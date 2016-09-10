clear; close all; clc

y = audioread('mail.wav');

plot(y);

for wsize = [512, 1024, 2048],
    spectrogram(y, wsize)
    pause;
end

S = spectrogram(y, 512);
S_res = abs(S(1:100, :));

imagesc(flipud(S_res));

% set(gca,'YDir','normal');  % can also invert the y aixs
    
