import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from typing import Tuple, Optional
from utils.memory import NavigationMemory
import math
from scipy.ndimage import gaussian_filter

def generate_occupancy_map(
    memory: NavigationMemory,
    camera_height: float = 1.5,  # meters
    camera_fov: float = 90.0,    # degrees
    camera_pitch: float = 15.0,  # degrees down from horizontal
    cell_size: float = 0.05,      # meters per cell
    map_size: Tuple[int, int] = (200, 200),  # cells (height, width)
    display: bool = True,
    save_path: Optional[str] = None,
    existing_map: Optional[np.ndarray] = None,
    start_position: Optional[Tuple[int, int]] = None,
    position_history: Optional[list] = None
) -> Tuple[np.ndarray, list]:
    """
    Generate a 2D occupancy map from navigation memory.
    
    Args:
        memory: NavigationMemory instance containing movement history and depth images
        camera_height: Height of the camera from the ground in meters
        camera_fov: Field of view of the camera in degrees
        camera_pitch: Camera pitch angle in degrees (positive is looking down)
        cell_size: Size of each cell in the occupancy map in meters
        map_size: Size of the occupancy map in cells (height, width)
        display: Whether to display the map
        save_path: Path to save the map image
        existing_map: Optional existing occupancy map to update (preserves history)
        start_position: Optional starting position (x, y) if different from center
        position_history: Optional existing position history to continue from
        
    Returns:
        Tuple containing (updated occupancy map as a numpy array, position history)
    """
    # Initialize or reuse occupancy map (0 = unknown, 1 = free, 2 = occupied)
    if existing_map is not None:
        occupancy_map = existing_map
    else:
        occupancy_map = np.zeros(map_size, dtype=np.uint8)
    
    # Start position at the center of the map or provided position
    if start_position is not None:
        start_x, start_y = start_position
    else:
        start_x, start_y = map_size[1] // 2, map_size[0] // 2
    
    # Initialize or continue position history
    if position_history is not None:
        position_history = position_history.copy()  # Create a copy to avoid modifying the original
        current_x, current_y = position_history[-1][0], position_history[-1][1]
        last_step_number = position_history[-1][2]
    else:
        position_history = [(start_x, start_y, 0)]  # (x, y, step_number)
        current_x, current_y = start_x, start_y
        last_step_number = 0
    
    # Initialize orientation based on existing history
    if len(position_history) > 1:
        # Calculate orientation from last two positions
        prev_x, prev_y = position_history[-2][0], position_history[-2][1]
        dx = current_x - prev_x
        dy = current_y - prev_y
        current_orientation = math.degrees(math.atan2(dy, dx)) % 360
    else:
        current_orientation = 90  # 90 degrees = facing up (north)
    
    # Mark start position as free if it's a new map
    if existing_map is None:
        occupancy_map[start_y, start_x] = 1
    
    # Only process actions that we haven't processed before
    start_idx = 0
    if last_step_number > 0:
        # Find where to continue from
        for i, action in enumerate(memory.action_memory):
            if action.step_number > last_step_number:
                start_idx = i
                break
    
    # Process each action in memory starting from where we left off
    for i in range(start_idx, len(memory.action_memory)):
        action = memory.action_memory[i]
        step_number = action.step_number
        
        # Process depth image and navigability mask BEFORE applying movement
        # Since observations at index i come before action at index i
        if i < len(memory.depth_memory) and i < len(memory.navigability_masks):
            depth_image = memory.depth_memory[i]
            navigability_mask = memory.navigability_masks[i]
            update_occupancy_from_depth(
                occupancy_map, 
                depth_image,
                navigability_mask,
                (current_x, current_y),
                current_orientation,
                camera_height, 
                camera_fov, 
                camera_pitch,
                cell_size,
                map_size
            )
        
        # Now handle the movement based on the action
        if action.movement_info:
            action_type = action.movement_info.get('action')
            
            # Handle rotation actions
            if action_type == 'RotateLeft':
                current_orientation = (current_orientation - action.movement_info.get('degrees', 0)) % 360
            elif action_type == 'RotateRight':
                current_orientation = (current_orientation + action.movement_info.get('degrees', 0)) % 360
            elif action_type == 'TurnAround':
                current_orientation = (current_orientation + 180) % 360
            
            # Handle movement
            if 'move_distance' in action.movement_info and action.is_success:
                distance = action.movement_info['move_distance']
                # Convert to grid cells
                move_cells = int(distance / cell_size)
                
                # Calculate movement in x, y coordinates
                dx = move_cells * math.cos(math.radians(current_orientation))
                dy = move_cells * math.sin(math.radians(current_orientation))
                
                # Update position
                current_x = int(current_x + dx)
                current_y = int(current_y + dy)
                
                # Ensure we stay within map bounds
                current_x = max(0, min(current_x, map_size[1] - 1))
                current_y = max(0, min(current_y, map_size[0] - 1))
                
                # Mark path as free
                # Draw a line from previous position to current position
                prev_x, prev_y = position_history[-1][0], position_history[-1][1]
                
                # Use Bresenham's line algorithm to mark cells along the path
                line_cells = bresenham_line(prev_x, prev_y, current_x, current_y)
                for x, y in line_cells:
                    if 0 <= x < map_size[1] and 0 <= y < map_size[0]:
                        # Only mark as free if not already marked as obstacle
                        if occupancy_map[y, x] != 2:
                            occupancy_map[y, x] = 1  # Mark as free
                
                # Add current position to history
                position_history.append((current_x, current_y, step_number))
    
    # Apply Kalman filtering to improve the occupancy map
    filtered_map = apply_kalman_filter(occupancy_map)
    
    if display or save_path:
        visualize_occupancy_map(
            filtered_map, 
            position_history,
            start_x, 
            start_y,
            current_x,
            current_y,
            current_orientation,
            camera_fov,
            cell_size,
            display,
            save_path
        )
    
    return filtered_map, position_history

def update_occupancy_from_depth(
    occupancy_map: np.ndarray,
    depth_image: np.ndarray,
    navigability_mask: np.ndarray,
    agent_pos: Tuple[int, int],
    orientation: float,
    camera_height: float,
    camera_fov: float,
    camera_pitch: float,
    cell_size: float,
    map_size: Tuple[int, int]
):
    """
    Update occupancy map using depth image and navigability mask information.
    
    Args:
        occupancy_map: Current occupancy map
        depth_image: Depth image from agent's perspective
        navigability_mask: Mask showing navigable (True) vs obstacle (False) cells
        agent_pos: Agent's position (x, y) in grid coordinates
        orientation: Agent's orientation in degrees
        camera_height: Height of the camera from the ground
        camera_fov: Field of view of the camera in degrees
        camera_pitch: Camera pitch in degrees
        cell_size: Size of each cell in meters
        map_size: Size of the occupancy map in cells
    """
    if depth_image is None or depth_image.size == 0:
        return
    
    if navigability_mask is None or navigability_mask.size == 0:
        return
    
    # Get agent position
    agent_x, agent_y = agent_pos
    
    # Get depth image dimensions
    height, width = depth_image.shape[:2]
    
    # Calculate horizontal and vertical FOV
    aspect_ratio = width / height
    h_fov = camera_fov  # Horizontal FOV
    v_fov = h_fov / aspect_ratio  # Vertical FOV
    
    # Create a temporary map of cells seen in this observation
    seen_cells = set()
    
    # Process each pixel in the depth image (subsample for efficiency)
    stride = 6  # Process every Nth pixel (reduced from 8 to capture more detail)
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            depth = depth_image[y, x]
            
            # Skip invalid depth values
            if depth <= 0 or np.isnan(depth) or depth > 10.0:  # Ignore very far readings
                continue
            
            # Get navigability for this pixel
            is_navigable = navigability_mask[y, x] if y < navigability_mask.shape[0] and x < navigability_mask.shape[1] else False
            
            # Convert to real-world distance in meters
            distance = depth
            
            # Calculate angle offsets from center of view
            angle_h = ((x / width) - 0.5) * h_fov
            angle_v = ((y / height) - 0.5) * v_fov
            
            # Adjust for camera pitch
            adjusted_angle_v = angle_v - camera_pitch
            
            # Calculate 3D position relative to camera
            # X is right, Y is up, Z is forward in camera space
            z = distance * math.cos(math.radians(adjusted_angle_v))
            y_offset = distance * math.sin(math.radians(adjusted_angle_v))
            x_offset = z * math.tan(math.radians(angle_h))
            
            # Calculate ground distance (in camera space)
            ground_distance = math.sqrt(z**2 + x_offset**2)
            
            # Skip points that are too close to the agent
            if ground_distance < 0.1:
                continue
                
            # Convert to world space coordinates and then to grid coordinates
            world_angle = (orientation + angle_h) % 360
            world_x = agent_x + (ground_distance * math.cos(math.radians(world_angle)) / cell_size)
            world_y = agent_y + (ground_distance * math.sin(math.radians(world_angle)) / cell_size)
            
            grid_x = int(world_x)
            grid_y = int(world_y)
            
            # Check if within map bounds
            if 0 <= grid_x < map_size[1] and 0 <= grid_y < map_size[0]:
                # Add to seen cells
                seen_cells.add((grid_x, grid_y))
                
                # Mark cell based on navigability
                if not is_navigable:
                    # Use a confidence-based approach for obstacles
                    if occupancy_map[grid_y, grid_x] == 2:
                        # Already marked as obstacle, keep it
                        pass
                    elif occupancy_map[grid_y, grid_x] == 1:
                        # Previously free, but now observed as obstacle
                        # Only mark as obstacle if we're confident (close enough)
                        if distance < 3.0:  # Only trust obstacle detection within 3 meters
                            occupancy_map[grid_y, grid_x] = 2
                    else:
                        # Unknown before, mark as obstacle
                        occupancy_map[grid_y, grid_x] = 2
                else:
                    # For navigable areas, allow updates from obstacle to free 
                    # but only when we have clear, close observations
                    if occupancy_map[grid_y, grid_x] == 2:
                        # Previously obstacle, now looks navigable
                        # Update only if we're very confident (very close observation)
                        if distance < 2.0:  # Only update obstacles within 2 meters
                            occupancy_map[grid_y, grid_x] = 1
                    else:
                        # Unknown or already free, mark as free
                        occupancy_map[grid_y, grid_x] = 1
    
    # Mark cells between agent and observed cells
    for grid_x, grid_y in seen_cells:
        line_cells = bresenham_line(agent_x, agent_y, grid_x, grid_y)
        for i, (lx, ly) in enumerate(line_cells[:-1]):  # All except the last point
            if 0 <= lx < map_size[1] and 0 <= ly < map_size[0]:
                # Allow more dynamic updating based on confidence
                if occupancy_map[ly, lx] == 2:
                    # Only update obstacles along line of sight with caution
                    # Keep obstacle classification unless we've seen this area multiple times
                    point_distance = math.sqrt((lx - agent_x)**2 + (ly - agent_y)**2) * cell_size
                    if point_distance < 1.5:  # Only update obstacles along line of sight if close
                        occupancy_map[ly, lx] = 1  # Update to free if very close
                else:
                    occupancy_map[ly, lx] = 1  # Mark as free
    
    # Mark the immediate area around the agent as free (improved local area mapping)
    local_radius = int(1.0 / cell_size)  # 1 meter radius
    for dx in range(-local_radius, local_radius + 1):
        for dy in range(-local_radius, local_radius + 1):
            nx, ny = agent_x + dx, agent_y + dy
            if 0 <= nx < map_size[1] and 0 <= ny < map_size[0]:
                # Check if within circle
                if dx*dx + dy*dy <= local_radius*local_radius:
                    # Calculate angle to this cell from agent position
                    cell_angle = math.degrees(math.atan2(dy, dx)) % 360
                    # Convert agent orientation to 0-360 format for comparison
                    agent_orientation = orientation % 360
                    # Calculate angular difference, accounting for wrap-around
                    angle_diff = min((cell_angle - agent_orientation) % 360, 
                                    (agent_orientation - cell_angle) % 360)
                    # Check if within FOV
                    if angle_diff <= camera_fov / 2:
                        # Don't overwrite occupied cells
                        if occupancy_map[ny, nx] != 2:
                            occupancy_map[ny, nx] = 1  # Free

def bresenham_line(x0, y0, x1, y1):
    """
    Generate points in a line from (x0, y0) to (x1, y1) using Bresenham's algorithm.
    
    Args:
        x0, y0: Starting coordinates
        x1, y1: Ending coordinates
        
    Returns:
        List of (x, y) coordinates along the line
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
    return points

def apply_kalman_filter(occupancy_map: np.ndarray) -> np.ndarray:
    """
    Apply a simplified Kalman filter to improve the appearance of the occupancy map.
    
    This implementation uses Gaussian filtering as a simplified approach to Kalman
    filtering for spatial data, which helps smooth the occupancy grid while
    preserving important features.
    
    Args:
        occupancy_map: The original occupancy map
        
    Returns:
        Filtered occupancy map
    """
    # Create a copy of the map for filtering
    filtered_map = occupancy_map.copy().astype(float)
    
    # Extract different regions
    unknown_region = (occupancy_map == 0).astype(float)
    free_region = (occupancy_map == 1).astype(float)
    occupied_region = (occupancy_map == 2).astype(float)
    
    # Apply Gaussian smoothing to each region separately with optimized parameters
    # This preserves the distinction between different types of cells
    smoothed_free = gaussian_filter(free_region, sigma=0.8)  # Reduced sigma for more defined free space
    smoothed_occupied = gaussian_filter(occupied_region, sigma=0.5)  # Reduced sigma for more defined obstacles
    
    # Apply a lighter filter to unknown regions to reduce ambiguity but avoid overclassification
    smoothed_unknown = gaussian_filter(unknown_region, sigma=1.0)  # Reduced from 1.5
    
    # Combine the smoothed regions with adjusted thresholds
    filtered_map = np.zeros_like(occupancy_map)
    
    # Priority: occupied > free > unknown
    # Less conservative thresholds for occupied areas
    filtered_map[smoothed_occupied > 0.3] = 2  # Lowered from 0.35
    
    # Be more aggressive with free space classification
    free_mask = np.logical_and(filtered_map == 0, smoothed_free > 0.15)  # Lowered from 0.2
    free_mask = np.logical_and(free_mask, smoothed_occupied < 0.1)  # Increased from 0.08
    filtered_map[free_mask] = 1  # Free cells
    
    # Fill in small gaps of unknown cells surrounded by free cells - more conservative
    for _ in range(1):  # Apply only once (was twice)
        unknown_mask = (filtered_map == 0)
        unknown_indices = np.where(unknown_mask)
        
        for i in range(len(unknown_indices[0])):
            y, x = unknown_indices[0][i], unknown_indices[1][i]
            
            # Check 8-connected neighbors
            neighbors_free = 0
            neighbors_occupied = 0
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                        
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < filtered_map.shape[0] and 0 <= nx < filtered_map.shape[1]:
                        if filtered_map[ny, nx] == 1:
                            neighbors_free += 1
                        elif filtered_map[ny, nx] == 2:
                            neighbors_occupied += 1
            
            # Only fill if strongly surrounded by free cells
            if neighbors_free >= 5 and neighbors_occupied == 0:  # Reduced from 6
                filtered_map[y, x] = 1
    
    # Don't automatically preserve all original obstacles - allow more updates
    # but keep high confidence obstacles from the original map
    obstacle_confidence_mask = (occupancy_map == 2) & (smoothed_occupied > 0.4)
    filtered_map[obstacle_confidence_mask] = 2
    
    # Convert to uint8 for consistency with the original map
    return filtered_map.astype(np.uint8)

def visualize_occupancy_map(
    occupancy_map: np.ndarray,
    position_history: list,
    start_x: int,
    start_y: int,
    current_x: int,
    current_y: int,
    current_orientation: float,
    camera_fov: float,
    cell_size: float,
    display: bool = True,
    save_path: Optional[str] = None
):
    """
    Visualize the occupancy map with path history and agent orientation.
    
    Args:
        occupancy_map: Occupancy map (0=unknown, 1=free, 2=occupied)
        position_history: List of agent positions with step numbers [(x, y, step)]
        start_x, start_y: Starting position
        current_x, current_y: Current position
        current_orientation: Current orientation in degrees
        camera_fov: Field of view in degrees
        cell_size: Size of each cell in meters
        display: Whether to display the map
        save_path: Path to save the map image
    """
    # Create a color map for visualization - enhanced colors for better visibility
    cmap = plt.cm.colors.ListedColormap(['#DDDDDD', '#FFFFFF', '#333333'])  # Light gray for unknown, White for free, Dark gray for obstacles
    
    # Create figure and axes explicitly
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Plot the occupancy map and store the image object
    im = ax.imshow(occupancy_map, cmap=cmap, origin='lower', interpolation='nearest')
    
    # Flip the x-axis to correct left-right orientation
    ax.invert_xaxis()
    
    # Plot path with improved visualization
    path_x = [x for x, y, _ in position_history]
    path_y = [y for x, y, _ in position_history]
    
    # Draw agent trajectory with gradient color to show direction
    points = np.array([path_x, path_y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Use a colormap to show progression of movement
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create custom colormap for the path
    cmap_path = LinearSegmentedColormap.from_list('path_cmap', ['blue', 'red'])
    norm = plt.Normalize(0, len(segments))
    lc = LineCollection(segments, cmap=cmap_path, norm=norm, linewidth=2, alpha=0.8)
    lc.set_array(np.arange(len(segments)))
    ax.add_collection(lc)
    
    # Add markers for important positions
    # Start position
    ax.plot(start_x, start_y, 'go', markersize=10, label='Start Position')
    # Current position
    ax.plot(current_x, current_y, 'ro', markersize=10, label='Current Position')
    
    # Add intermediate waypoints, but less frequently for clarity
    step_interval = max(1, len(position_history) // 8)  # Show about 8 waypoints
    for i in range(0, len(position_history), step_interval):
        x, y, step = position_history[i]
        if i > 0 and i < len(position_history) - 1:  # Skip start and end which are already plotted
            ax.plot(x, y, 'bo', markersize=5)
            ax.text(x+2, y+2, f"{step}", fontsize=8, color='blue')
    
    # Draw agent's current view direction as a circular sector
    current_wedge = Wedge(
        (current_x, current_y), 
        12,  # radius
        current_orientation - camera_fov/2, 
        current_orientation + camera_fov/2, 
        alpha=0.5, 
        color='yellow',
        label='Current View'
    )
    ax.add_patch(current_wedge)
    
    # Add legend and labels
    ax.set_title('Occupancy Map with Navigation History', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add colorbar with improved labels
    cbar = fig.colorbar(im, ax=ax, ticks=[0.16, 0.5, 0.83], fraction=0.025, pad=0.04)
    cbar.ax.set_yticklabels(['Unknown', 'Navigable', 'Obstacle'], fontsize=9)
    
    # Add direction indicators and map info
    ax.text(10, 10, f"Map size: {occupancy_map.shape[1]}×{occupancy_map.shape[0]} cells, {cell_size:.3f}m per cell", 
            fontsize=9, bbox=dict(facecolor='white', alpha=0.7))
    
    # Add map scale
    scalebar_length = 20  # cells
    ax.plot([10, 10 + scalebar_length], [5, 5], 'k-', linewidth=2)
    ax.text(10 + scalebar_length/2, 3, f"{scalebar_length * cell_size:.2f}m", 
            horizontalalignment='center', fontsize=8)
    
    # Add step count information
    if position_history:
        last_step = position_history[-1][2]
        ax.text(10, occupancy_map.shape[0] - 10, f"Steps: {last_step}", 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if display:
        plt.show()
    else:
        plt.close(fig)
