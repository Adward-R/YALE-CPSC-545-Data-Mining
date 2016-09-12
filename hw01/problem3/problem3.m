clear; close all; clc

y = audioread('mail.wav');
plot(y);
print('plot_wave', '-dpng');

for wsize = [512, 1024, 2048],
    spectrogram(y, wsize);
    print(strcat('spec_', int2str(wsize)), '-dpng');
    % pause;
end

S = spectrogram(y, 512);
S_res = abs(S(1:100, :));

imagesc(flipud(S_res));
print('flipud_s', '-dpng');

% set(gca,'YDir','normal');  % can also invert the y aixs
    
