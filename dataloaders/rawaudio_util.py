import soundfile as sf
import numpy as np
import librosa
import pdb
import random
import warnings

warnings.filterwarnings('ignore')


def time_warp(spec, W=5):
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    # assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]
    # assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)

def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram[0, :, :], ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def spec_augment(mel_spectrogram, time_warping_para=0, frequency_masking_para=10,
                 time_masking_para=10, frequency_mask_prob=0.05, time_mask_prob=0.15):
    """Spec augmentation Calculation Function.
    'SpecAugment' have 3 steps for audio data augmentation.
    first step is time warping using Tensorflow's image_sparse_warp function.
    Second step is frequency masking, last step is time masking.
    # Arguments:
      mel_spectrogram(numpy array): audio file path of you want to warping and masking.
      time_warping_para(float): Augmentation parameter, "time warp parameter W".
        If none, default = 80 for LibriSpeech.
      frequency_masking_para(float): Augmentation parameter, "frequency mask parameter F"
        If none, default = 100 for LibriSpeech.
      time_masking_para(float): Augmentation parameter, "time mask parameter T"
        If none, default = 27 for LibriSpeech.
      frequency_mask_num(float): number of frequency masking lines, "m_F".
        If none, default = 1 for LibriSpeech.
      time_mask_num(float): number of time masking lines, "m_T".
        If none, default = 1 for LibriSpeech.
    # Returns
      mel_spectrogram(numpy array): warped and masked mel spectrogram.
    """
    l = mel_spectrogram.shape[1]
    h = mel_spectrogram.shape[2]

    # Step 1 : Time warping
    if time_warping_para > 0:
        mel_spectrogram = time_warp(mel_spectrogram, W=time_warping_para)

    # Step 2 : Frequency masking
    frequency_mask_num = int(h * frequency_mask_prob)
    for i in range(frequency_mask_num):
        f = np.random.uniform(low=0.0, high=frequency_masking_para)
        f = int(f)
        f0 = random.randint(0, h-f)
        mel_spectrogram[:, :, f0:f0+f] = 0

    # Step 3 : Time masking
    time_mask_num = int(l * time_mask_prob)
    for i in range(time_mask_num):
        t = np.random.uniform(low=0.0, high=time_masking_para)
        t = int(t)
        t0 = random.randint(0, l-t)
        mel_spectrogram[:, t0:t0+t,:] = 0

    return mel_spectrogram

def audio_processor(audio_frames):
    '''
    normalize the raw audio_frames:[len,3,224,224]
    '''
    # mean=(0.48145466, 0.4578275, 0.40821073),
    
    # std=(0.26862954, 0.26130258, 0.27577711)
    mu = np.mean(audio_frames, axis=(2,3))[:, :, None, None]
    sigma = np.std(audio_frames, axis=(2,3))[:, :, None, None] + 1e-10
    audio_frames = (audio_frames - mu)/sigma
    return audio_frames
    
def get_raw_audio(audio_path,target_rate, start=None, end=None, max_s=60):
   
    if start is not None and end is not None:
        start_f,  end_f = target_rate*start, target_rate*end
        audio_wav, audio_rate = sf.read(audio_path,start=start_f, stop=end_f)
    else:
        audio_wav, audio_rate = sf.read(audio_path)    
   
    
    
    

    max_audio_len = max_s * audio_rate
    if audio_wav.shape[0] > max_audio_len:
      start = int(audio_wav.shape[0]/2 - max_audio_len/2)
      end = int( start + max_audio_len)
      audio_wav = audio_wav[start:end,:]

    if target_rate < audio_rate:
      #down sampling
      audio_wav = librosa.resample(audio_wav, audio_rate, target_rate)
    
    if audio_wav.ndim == 1: 
        audio_wav = np.repeat(audio_wav[:, np.newaxis], 2, axis=1)


    return audio_wav


def wav2fbank(waveform, audio_rate):
    # get wavform and turn to fbank every audio frame 224*224
    # waveform:[1,]
    
    melspec= librosa.feature.melspectrogram(waveform, audio_rate, n_fft=512, hop_length=128, n_mels=224, fmax=8000)
    logmelspec = librosa.power_to_db(melspec)

    
    return logmelspec.T

def split_frame(audio_frame, overlap=0, single_frame_len=224):
     
    audio_pad = single_frame_len - audio_frame.shape[1] % single_frame_len
           
    if audio_pad > 0:
        zero_pad = np.zeros([3, audio_pad, single_frame_len])
        audio_frame = np.concatenate([audio_frame, zero_pad], axis=1)
    audio_frame = audio_frame.reshape(3, -1, single_frame_len, single_frame_len).transpose(1,0,2,3)
    return audio_frame


# if __name__=='__main__':
#     import matplotlib.pyplot as plt
#     audio_path = '/dataset/28d47491/rld/data/msrvtt/audios_16k/video284.wav'
    
#     audio_wav = get_raw_audio(audio_path, 16000)
#     audio_frame_l = wav2fbank(audio_wav[:,0],16000)
#     audio_frame_r = wav2fbank(audio_wav[:,1],16000)
#     audio_frame_m = wav2fbank((audio_wav[:, 0]+audio_wav[:, 1])/2,16000)

#     pdb.set_trace()
#     audio_frame = np.stack([audio_frame_l, audio_frame_m, audio_frame_r], axis=0)
   
    
#     audio_frame_aug = spec_augment(audio_frame)
#     audio_frame_aug = split_frame(audio_frame, overlap=0, single_frame_len=224)
#     audio_frame_aug = audio_processor(audio_frame_aug)

#     audio_frame = split_frame(audio_frame, overlap=0, single_frame_len=224)
#     audio_frame = audio_processor(audio_frame)


    
