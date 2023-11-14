import librosa
import scipy
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

input_file = 'some_audio.wav'
output_file = 'output_audio.wav'
figsize = (10, 5)


def main():
    audio, samplerate = librosa.load(input_file)
    noise = audio[:samplerate]

    output = remove_noise(audio=audio, noise=noise)
    sf.write(output_file, output, samplerate)
    
    plot_audio(audio, samplerate, title='Input audio')
    plot_audio(output, samplerate, title='Output audio')
    
    plt.show()


def remove_noise(audio, noise, kernel_size=5, std_threshold=1.5):
    audio_stft = librosa.stft(audio)
    noise_stft = librosa.stft(noise)
    
    audio_db = librosa.amplitude_to_db(np.abs(audio_stft))
    noise_db = librosa.amplitude_to_db(np.abs(noise_stft))

    mean_freq_noise = np.mean(noise_db, axis=1)
    std_freq_noise = np.std(noise_db, axis=1)
    noise_threshold = mean_freq_noise + std_freq_noise * std_threshold

    db_thresh = np.repeat(noise_threshold[:, np.newaxis], audio_stft.shape[1], axis=1)
    mask = audio_db < db_thresh

    kernel = np.outer(
        np.hanning(kernel_size)[1:-1], 
        np.hanning(kernel_size)[1:-1]
    )
    kernel = kernel / np.sum(kernel)
    mask = np.minimum(1, scipy.signal.fftconvolve(mask, kernel, mode="same"))

    denoised_stft = audio_stft * (1 - mask)
    denoised_db = librosa.amplitude_to_db(np.abs(denoised_stft))
    denoised_audio = librosa.istft(denoised_stft)

    plot_spectrogram(audio_db, title="Input audio")
    plot_spectrogram(denoised_db, title="Output audio")

    return denoised_audio


def plot_spectrogram(signal, title=None):
    plt.figure(figsize=figsize)
    plt.imshow(signal, origin="lower", aspect="auto", cmap=plt.cm.seismic,
               vmin=-np.max(np.abs(signal)), vmax=np.max(np.abs(signal)))
    plt.colorbar()
    if title:
        plt.title(title)
    # plt.tight_layout()


def plot_audio(signal, sr=None, title=None):
    plt.figure(figsize=figsize)
    
    t = np.arange(0, signal.shape[0], dtype=np.float64)
    if sr:
        t /= float(sr)
    
    plt.plot(t, signal)
    
    if title:
        plt.title(title)


if __name__ == '__main__':
    main()
