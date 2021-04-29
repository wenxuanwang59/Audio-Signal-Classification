
%-------------------Part 1: Plot signal and save---------------------------
clear all;
% read the audio signal,extract the middle 0.6s ; plot and save
signal_length = round(44100 * 0.6);
[ ss,fs ] = audioread ('recording2/s1A.wav');
s = middle_signal(ss,signal_length);
time = 1 : signal_length;
figure(1)
plot ( time , s );
hold on
xlabel (' time n ')
ylabel (' Signal S ')
title('Signal Wave')
axis([1 signal_length -0.6 0.6])
ylim=get(gca,'Ylim');
saveas ( gcf,'s1A.jpg' );


%------------Part 2: Detect the signal start and end point-----------------

% Frameblocking
% set every frame = 441sample£¬overlapping 0%
frame_size = 441;
overlap = 0.0;
frame_step = frame_size*(1-overlap);
frame_num = floor(signal_length/frame_step);
% store each frame in a 2-dimension array
frame_array = zeros(frame_num,frame_size);       

%%% calculate the energy of each frame
energy = zeros(frame_num);
for i = 1:frame_num
    frame_array(i,:)=s(frame_step*(i-1)+1:frame_step*(i-1)+frame_size);
    energy(i)= sum(frame_array(i,:).^2);
end
% normalize energy to [0,1]
energy = energy/max(energy);
figure(2)
plot(1:frame_num , energy);
xlabel (' frame ')
ylabel (' Energy ')
title('Frame energy')


%%% calculate the zero crossing rate of each frame
zerocross = zeros(frame_num);
for i = 1 : frame_num
    for j = 1:frame_size-1
        zerocross(i) = zerocross(i)+(abs(sign(frame_array(i,j))-sign(frame_array(i,j+1))))/2;
    end
end
% normalize zerocross to [0,1]
zerocross = zerocross/max(zerocross);
figure(3)
plot(1:frame_num , zerocross);
xlabel (' frame ')
ylabel (' Zero Crossing Rate ')
title('Frame zero crossing rate')

%%% detect the start point
energy_high = 0.4;
energy_low = 0.1;
zero_high = 0.05;
zero_low = 0.05;
start_frame = 1;
% if 3 successive frame has high energy and high zero crossing rate , it is
% considered as the start point.
for i = 1: frame_num-2
    if (energy(i)> energy_high && energy(i+1)>energy_high && energy(i+2)>energy_high)
        if (zerocross(i)>zero_high && zerocross(i+1)>zero_high && zerocross(i+2)>zero_high)
            start_frame = i;
            fprintf('the start point is near ');
            fprintf('%d',round(start_frame * frame_step/441*10));
            fprintf(' ms\n');
            % mark the approximate start point in the original signal
            figure(1)
            x=start_frame * frame_step;
            plot([x,x],ylim,'m--');
            break
        end
    end
end


%%% detect the end point
% if 5 successive frame has low energy and low zero crossing rate , it is
% considered as the end point.
for i = start_frame: frame_num-4
    if (energy(i)< energy_low && energy(i+1)<energy_low && energy(i+2)<energy_low && energy(i+3)<energy_low && energy(i+4)<energy_low)
        if (zerocross(i)<zero_low && zerocross(i+1)<zero_low && zerocross(i+2)<zero_low && zerocross(i+3)<zero_low && zerocross(i+4)<zero_low)
            end_frame = i;
            fprintf('the end point is near ');
            fprintf('%d',round(end_frame * frame_step/441*10));
            fprintf(' ms\n');
            % mark the approximate end point in the original signal
            figure(1)
            x=end_frame * frame_step;
            plot([x,x],ylim,'m--');
            break
        end
    end
end


%-----------------Part 3 : do DFT to a 20ms segment------------------------

% plot the original signal wave in time domain
seg1_size = (20 / 1000) * 44100;
seg1_start = frame_step * ( start_frame + 5 );
seg1 = zeros(seg1_size);
for i = 1:seg1_size
     seg1(i) = s(seg1_start+i);
end
figure(4)
subplot(2,1,1)
plot(1:seg1_size,seg1)
xlabel (' time ')
ylabel (' signal ')
title('seg1 in time domain')

% do DFT to seg1 and plot in frequency domain
a=zeros(1,seg1_size / 2);
b=zeros(1,seg1_size / 2);
c = zeros(1,seg1_size / 2);
for k = 1 : seg1_size / 2 
    for n = 1 : seg1_size
        a(k)=a(k)+seg1(n)*cos(2*pi*k*n/seg1_size);
        b(k)=b(k)+seg1(n)*sin(2*pi*k*n/seg1_size);
    end
c(k)=sqrt(a(k)^2+b(k)^2);
end

subplot(2,1,2)
plot(1:seg1_size / 2,c);
title('seg1 in frequency domain');
xlabel('Frequency');
ylabel('Energy');
saveas ( gcf,'Fourier_seg1.jpg');

%-----------------Part 4 : Pre-emphasize-----------------------------------

alpha = 0.945;
figure(5)
subplot(2,1,1)
plot(1:seg1_size,seg1)
title('seg1 wave');
xlabel('time');
ylabel('signal');

pem_seg1 = zeros(seg1_size - 1);
for n = 1:seg1_size - 1
     pem_seg1(n) = seg1(n+1) - alpha * seg1(n);
end
subplot(2,1,2)
plot(1:seg1_size - 1,pem_seg1)
title('pre-emphasized seg1 wave');
xlabel('time');
ylabel('signal');

%-----------------Part 5 : Find the 10-order LPC parameters----------------

order = 10;
r = zeros(order+1);
for i=1:order+1
  r(i)=0.0;
  for j=1:seg1_size-i+1
     r(i) = r(i)+pem_seg1(j) * pem_seg1(j+i-1);
  end
end

m = zeros(order,order);
for i=1:order
  for j=1:order
        if j >= i
          m(i,j) = r(j - i+1);
        else
          m(i,j) = r(i - j+1);
        end
  end
end
coef=inv(m)*reshape(r(2:order+1),order,1);
for i = 1:order
    fprintf('a')
    fprintf(num2str(i))
    fprintf(' is : ')
    fprintf('%f', coef(i))
    fprintf('\n')
end



%--------------------------End-------------------------------------------

%-------------------------Function part----------------------------------

%%% this function is used to crop the input signal and produce a
%%% specific-length output signal.
function [s] = middle_signal(ss,len)
s_start = floor(length(ss)/2-len/2);
s_end = s_start + len - 1;
s=ss(s_start:s_end);
end


