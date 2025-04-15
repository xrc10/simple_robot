import yaml
from models import VLM
from ai2thor.controller import Controller
import numpy as np
import cv2
import os
import shutil


class ThorEnv(object):
    def __init__(self, floor_id) -> None:
       if floor_id.startswith("FloorPlan_"):
           self.agentMode = "locobot"
       else:
           self.agentMode = "default"
       self.controller = Controller(
            agentMode=self.agentMode,
            visibilityDistance=1.5,
            scene=floor_id,

            # step sizes
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,

            # image modalities
            renderDepthImage=False,
            renderInstanceSegmentation=False,

            # camera properties
            width=300,
            height=300,
            fieldOfView=90
    )
       
    def get_last_event(self):
        return self.controller.last_event
       
    def step(self, action, degrees=30):
        if 'Rotate' in action:
            event = self.controller.step(action=action, degrees=degrees)
        elif 'Done' in action:
            event = self.get_last_event()
        else:
            event = self.controller.step(action=action)
        
        return event

class ThorEnvDogView(object):
    def __init__(self, floor_id) -> None:
        if floor_id.startswith("FloorPlan_"):
           self.agentMode = "locobot"
        else:
           self.agentMode = "default"
        self.controller = Controller(
            agentMode=self.agentMode,
            scene=floor_id,
            width=720,
            height=480,
            fieldOfView=80,

            # step sizes
            gridSize=0.25,
            snapToGrid=True,
            rotateStepDegrees=90,    

            # image modalities
            renderDepthImage=True,
            renderInstanceSegmentation=False,
        )
        if self.agentMode == "default": # only default can crouch
            self.controller.step("Crouch")
        self.controller.step("LookDown")
        self.camera_fov = self.controller.initialization_parameters['fieldOfView']
    
    def reset(self, floor_id):
        self.controller.reset(scene=floor_id)
        if self.agentMode == "default": # only default can crouch
            self.controller.step("Crouch")
        self.controller.step("LookDown")

    def get_last_event(self):
        return self.controller.last_event
    
    def step(self, action, degrees=30, magnitude=0.25):
        if 'Rotate' in action:
            event = self.controller.step(action=action, degrees=degrees)
        elif 'Move' in action:
            event = self.controller.step(action=action, moveMagnitude=magnitude)
        else:
            event = self.controller.step(action=action)
        
        return event

    def get_observation(self):
        return self.controller.last_event.frame
    
    def get_depth(self):
        return self.controller.last_event.depth_frame
    
    