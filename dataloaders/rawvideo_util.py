import torch as th
import numpy as np
from PIL import Image
# pytorch=1.7.1
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
# pip install opencv-python
import cv2, math
import pdb, logging, time
import os
# cv2.setNumThreads(0)
# cv2.ocl.setUseOpenCL(False) 
logger = logging.getLogger(__name__)

class RawVideoExtractorCV2():
    def __init__(self, centercrop=False, size=224, framerate=-1, ):
        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.transform = self._transform(self.size)
        cv2.setNumThreads(0) 

    def _transform(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def video_to_tensor(self, video_file, preprocess, sample_fp=0, start_time=None, end_time=None):
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1
       
        # Samples a frame sample_fp X frames.
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
    
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        
        
        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1
        
        inds = [ind for ind in np.arange(0, fps, interval)]
        
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []

        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                cap.set(cv2.CAP_PROP_POS_FRAMES, sec_base + ind)
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(preprocess(Image.fromarray(frame_rgb).convert("RGB")))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return video_data

    def get_video_data(self, video_path, start_time=None, end_time=None):
        image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        return image_input

    def get_video_frames(self, frame_dir, start_time=None, end_time=None, video_fps=1, max_frames=12, frame_pos=2):
        video_duration = math.ceil(len(os.listdir(frame_dir))/video_fps)
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = video_duration
        else:
            end_time= min(video_duration, end_time)

        images = []
        times = list(range(start_time, end_time))
        if max_frames < len(times):
            if frame_pos == 0:
                times = times[:max_frames]
            elif frame_pos == 1:
                times = times[-max_frames:]
            else:
                sample_indx = np.linspace(0, len(times) - 1, num=max_frames, dtype=int)
                times = [times[i] for i in sample_indx]
                    

        for i in times:
            frame_path = os.path.join(frame_dir, str(i)+'.jpg')
            frame_rgb = cv2.imread(frame_path)
            if frame_rgb is None:
                continue
            images.append(self.transform(Image.fromarray(frame_rgb).convert("RGB")))
        
        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return video_data
     
    def extract_frames(self, video_file, sample_fp=0):
        
        cap = cv2.VideoCapture(video_file)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        
       
        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1
        
        inds = [ind for ind in np.arange(0, fps, interval)]
        
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]

        ret = True
        images, included = [], []
        
        for sec in np.arange(start_sec, end_sec + 1):
            if not ret: break
            sec_base = int(sec * fps)
            for ind in inds:
                frame_position = min(sec_base + ind, frameCount - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
                ret, frame = cap.read()
                if frame is None:
                    continue
                w, h, c = frame.shape
                m = min(w, h)
                ratio = self.size / m
                new_w, new_h = int(ratio * w), int(ratio *h)
                assert new_w > 0 and new_h > 0
                frame = cv2.resize(frame, (new_h, new_w))
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)# when training, needn't transfer channel
                images.append(frame_rgb)

        cap.release()

        
        return images

    def get_video_data_for_pre_decord(self, video_path, start_time=None, end_time=None, slice_framepos=2, max_frames=10):
        #image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        sample_fp = self.framerate
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1
    
        # Samples a frame sample_fp X frames.
        cap = decord.VideoReader(video_path, ctx=decord.cpu(0), width=self.size, height=self.size, num_threads=4)#cv2.VideoCapture(video_path)
        frameCount = len(cap)#int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get_avg_fps())#int(cap.get(cv2.CAP_PROP_FPS))
        try:
            total_duration = frameCount  // fps
            start_sec, end_sec = 0, total_duration
        except:
            print("error in {}".format(video_file) )
        
        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            #cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1
        
        inds = [ind for ind in np.arange(0, fps, interval)]# 0:3:3 frames every second picks 
         
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]
        picked_indexes = []

        for sec in np.arange(start_sec, end_sec + 1):
            sec_base = int(sec * fps)
            for ind in inds:
                picked_indexes.append(min(sec_base + ind, frameCount-1))
        
        if max_frames < len(picked_indexes):
            if slice_framepos == 0:
                picked_indexes = picked_indexes[:max_frames]
            elif slice_framepos == 1:
                picked_indexes = picked_indexes[-max_frames:, ...]
            else:
                sample_indx = np.linspace(0, len(picked_indexes) - 1, num=max_frames, dtype=int)
                picked_indexes = [picked_indexes[indx] for indx in sample_indx]

        ret = True
        
          
        frames = cap.get_batch(picked_indexes).asnumpy() #[len,224,224,3]
        del cap
        images=[]
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(self.transform(Image.fromarray(frame_rgb).convert("RGB")))
        
        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return {'video': video_data}
        
    def get_video_data_for_pre(self, video_path, start_time=None, end_time=None, slice_framepos=2, max_frames=10):
        #image_input = self.video_to_tensor(video_path, self.transform, sample_fp=self.framerate, start_time=start_time, end_time=end_time)
        global logger
        sample_fp = self.framerate
        if start_time is not None or end_time is not None:
            assert isinstance(start_time, int) and isinstance(end_time, int) \
                   and start_time > -1 and end_time > start_time
        assert sample_fp > -1
        
        # Samples a frame sample_fp X frames.
        get_cap_on = time.time()
        cap = cv2.VideoCapture(video_path)
        get_cap_off = time.time()-get_cap_on
        if get_cap_off > 10:
            print('read video in {} too long:{}'.format(video_path.split('/')[-1],get_cap_off ))
        
        get_info_on = time.time()
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        get_info_off = time.time() - get_info_on

        if get_info_off > 10:
            print('get video info in {} too long:{}'.format(video_path.split('/')[-1],get_info_off ))


        try:
            total_duration = (frameCount + fps - 1) // fps
            start_sec, end_sec = 0, total_duration
        except:
            print("error in {}".format(video_path) )
        
        if start_time is not None:
            start_sec, end_sec = start_time, end_time if end_time <= total_duration else total_duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))

        interval = 1
        if sample_fp > 0:
            interval = fps // sample_fp
        else:
            sample_fp = fps
        if interval == 0: interval = 1
        
        inds = [ind for ind in np.arange(0, fps, interval)]# 0:3:3 frames every second picks 
         
        assert len(inds) >= sample_fp
        inds = inds[:sample_fp]
        picked_indexes = []

        for sec in np.arange(start_sec, end_sec + 1):
            sec_base = int(sec * fps)
            for ind in inds:
                picked_indexes.append(sec_base + ind)
        
        if max_frames < len(picked_indexes):
            if slice_framepos == 0:
                picked_indexes = picked_indexes[:max_frames]
            elif slice_framepos == 1:
                picked_indexes = picked_indexes[-max_frames:, ...]
            else:
                sample_indx = np.linspace(0, len(picked_indexes) - 1, num=max_frames, dtype=int)
                picked_indexes = [picked_indexes[indx] for indx in sample_indx]

        ret = True
        images, included = [], []
        sample_frame_on = time.time()
        for picked_index in picked_indexes:
            if not ret: break
            cap.set(cv2.CAP_PROP_POS_FRAMES, picked_index)
            ret, frame = cap.read()
            if not ret: break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(self.transform(Image.fromarray(frame_rgb).convert("RGB")))
        sample_frame_off = time.time() - sample_frame_on
        if sample_frame_off > 10:
            print('sample video frame {} too long:{}'.format(video_path.split('/')[-1],sample_frame_off ))

        cap.release()

        if len(images) > 0:
            video_data = th.tensor(np.stack(images))
        else:
            video_data = th.zeros(1)
        return  video_data
        
        
    def process_raw_data(self, raw_video_data):
        tensor_size = raw_video_data.size()
        tensor = raw_video_data.view(-1, 1, tensor_size[-3], tensor_size[-2], tensor_size[-1])
        return tensor

    def process_frame_order(self, raw_video_data, frame_order=0):
        # 0: ordinary order; 1: reverse order; 2: random order.
        if frame_order == 0:
            pass
        elif frame_order == 1:
            reverse_order = np.arange(raw_video_data.size(0) - 1, -1, -1)
            raw_video_data = raw_video_data[reverse_order, ...]
        elif frame_order == 2:
            random_order = np.arange(raw_video_data.size(0))
            np.random.shuffle(random_order)
            raw_video_data = raw_video_data[random_order, ...]

        return raw_video_data

# An ordinary video frame extractor based CV2
RawVideoExtractor = RawVideoExtractorCV2

if __name__=='__main__':
    video_path = '../data/howto100m/videos_fps3/H3rQxlJm75c.mp4'
    
    RawVideoExtractor = RawVideoExtractorCV2(framerate=1)
    import time
    time_on=time.time()
    video_frames_decord = RawVideoExtractor.get_video_data_for_pre_decord(video_path,10,20)
    print('decord:{}'.format(time.time()-time_on))
    time_on = time.time()
    video_frames = RawVideoExtractor.get_video_data_for_pre(video_path,10,20)
    print('opencv:{}'.format(time.time()-time_on))