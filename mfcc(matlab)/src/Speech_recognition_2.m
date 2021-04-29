
%-------------------Part 0 : Read all files--------------------------------

%%% read all the audio signal from setA and setB, normalize their length to
%%% 0.7s.
signal_length = round(44100 * 0.7);
s = zeros(10,signal_length);

[ ss,fs ] = audioread ('recording2/s1A.wav');
s(1,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s5A.wav');
s(2,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s4A.wav');
s(3,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s6A.wav');
s(4,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s2A.wav');
s(5,:)=middle_signal(ss,signal_length);

[ ss,fs ] = audioread ('recording2/s1B.wav');
s(6,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s5B.wav');
s(7,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s4B.wav');
s(8,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s6B.wav');
s(9,:)=middle_signal(ss,signal_length);
[ ss,fs ] = audioread ('recording2/s2B.wav');
s(10,:)=middle_signal(ss,signal_length);

%--------------------Part 1 : convert into MFCC parameters------------

Tw = 10;           % analysis frame duration (ms)
Ts = 10;           % analysis frame shift (ms)
alpha = 0.945;     % preemphasis coefficient
R = [ 300 4000 ];  % frequency range to consider
M = 20;            % number of filterbank channels 
C = 13;            % number of cepstral coefficients
L = 22;            % cepstral sine lifter parameter
MFCC_matrix = zeros(10,13,70);

% hamming window (see Eq. (5.2) on p.73 of [1])
hamming = @(N)(0.54-0.46*cos(2*pi*[0:N-1].'/(N-1)));

% Feature extraction (feature vectors as columns)
for i = 1:10
    [ MFCCs, FBEs, frames ] = ...
                mfcc( s(i,:), fs, Tw, Ts, alpha, hamming, R, M, C, L );
    
    MFCC_matrix(i,:,:) = MFCCs;
end

%Plot cepstrum over time
figure('Position', [30 100 800 200], 'PaperPositionMode', 'auto', ... 
     'color', 'w', 'PaperOrientation', 'landscape', 'Visible', 'on' ); 

imagesc( [1:size(MFCCs,2)], [0:C-1], MFCCs ); 
axis( 'xy' );
xlabel( 'Frame index' ); 
ylabel( 'Cepstrum index' );
title( 'Mel frequency cepstrum' );


%-----------Part 2 : Compute the distortion matrix -----------------------
%------------------and the accumulated distance matrix--------------------


distortion_matrix = zeros(5,70,70);
accu_distance = zeros(5,70,70);
comparison_matrix = zeros(5,5);%% store the minimum accu_distance for each pair
minposition = zeros(5,2);%%store the position of the minimum accu_distance for each pair

for i = 1:5
    for m = 1 : 70
        for n = 1: 70
            distortion_matrix(i,m,n) = sqrt(sum((MFCC_matrix(i,2:13,m)-MFCC_matrix(i+5,2:13,n)).^2)); 
        end
    end
    %%start to calculate the accumulated distance
    accu_distance(i,1,1)=distortion_matrix(i,1,1);
    for n=2:70
        accu_distance(i,1,n)=distortion_matrix(i,1,n)+accu_distance(i,1,n-1);
    end
    
    for m=2:70
        accu_distance(i,m,1)=distortion_matrix(i,m,1)+accu_distance(i,m-1,1);
    end
    
    for m = 2 : 70
        for n = 2: 70
            accu_distance(i,m,n) = distortion_matrix(i,m,n) + min([accu_distance(i,m-1,n),accu_distance(i,m,n-1),accu_distance(i,m-1,n-1)]);
        end
    end
    %%start to calculate the minimum accu_distance and the position
    if min(accu_distance(i,70,:)) > min(accu_distance(i,:,70))
        [ comparison_matrix(i,i) , position ] = min(accu_distance(i,:,70));
        minposition(i,:) = [position,70];
    else
        [ comparison_matrix(i,i) , position ] = min(accu_distance(i,70,:));
        minposition(i,:) = [70,position];
    end
    
end

%%% print the comparison matrix table,save in excel file
comparison_matrix
xlswrite( 'comparison_matrix.xls', comparison_matrix)



%-----------Part 3 : Find optimal path in accumulated distance table-------

%%%find the optimal path using dynamic programming
t = 2;                     %%% pick the second pair sound 
path = zeros(140,2);       %% store the path information
path(1,:)=minposition(t,:);
m=minposition(t,1);
n=minposition(t,2);
for i = 2:140
   if accu_distance(t,m-1,n)== min([accu_distance(t,m-1,n),accu_distance(t,m,n-1),accu_distance(t,m-1,n-1)])
       m=m-1;
   elseif accu_distance(t,m,n-1)== min([accu_distance(t,m-1,n),accu_distance(t,m,n-1),accu_distance(t,m-1,n-1)])
       n=n-1;
   else
       m=m-1;
       n=n-1;
   end
   path(i,:)=[m,n];

   if m==1
       k=i;
       while n~=1
           k=k+1;
           n=n-1;
           path(k,:)=[1,n];
       end
       break;
   elseif n==1
       k=i;
       while m~=1
           k=k+1;
           m=m-1;
           path(k,:)=[m,1];
       end
       break;
   else
   end
end

for i= 1:size(path,1)
    if path(i,:)==[1,1]
        p_end=i;
        break
    end
end
path = path(1:p_end,:);

%%% plot the optimal path using UItable
display_path(reshape(accu_distance(t,:,:),70,70),path);


%--------------------------End-------------------------------------------

%-------------------------Function part----------------------------------

%%% this function is used to crop the input signal and produce a
%%% specific-length output signal.
function [s] = middle_signal(ss,len)
s_start = floor(length(ss)/2-len/2);
s_end = s_start + len - 1;
s=ss(s_start:s_end);
end

%%% this function is used to display the accumulated distance table in a
%%% figure , using the uitable tool.
%%% The optimal path is marked in red background.
function display_path(data,p)

data = mat2cell(data,ones(1,size(data,1)),ones(1,size(data,2)));

for i = 1:size(data,1)
    for j = 1:size(data,2)
        data{i,j}=['<html><table><tr><td width=100 >','<FONT face="Times New Roman"size="4"color=black">',num2str(data{i,j},'%10.2f'), '</table></html>'];
    end
end

for i = 1:size(p,1)
    data{p(i,1),p(i,2)}=['<html><table><tr><td width=100 bgcolor="red">','<FONT face="Times New Roman"size="4"color=black">',num2str(data{p(i,1),p(i,2)},'%10.2f'), '</table></html>'];
end

data2 = data(70:-1:1,:);
f2 = figure;
uitable(f2,'Data',data2,'Position',[20 20 1200 800]);
end
