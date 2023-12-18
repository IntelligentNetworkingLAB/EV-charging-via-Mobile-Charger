% 데이터 정의
EV1 = [173.264, 216.58];
EV2 = [154.938, 309.876];
EV3 = [179.095, 143.276];
EV4 = [268.226, 268.226];
EV5 = [269.892, 224.91];
EV6 = [143.276, 214.914];
EV7 = [170.765, 68.306];
EV8 = [127.449, 297.381];
Mean = [185.863, 218];
Std = [50.5, 75.26];

data = [EV1;EV2;EV3;EV4;EV5;EV6;EV7;EV8;Mean;Std];
% 그래프 그리기
figure;
bar(data, 'grouped');

% 그래프 제목과 레이블 추가
xlabel('EV');
ylabel('Distance');
% x축에 빨간 선 추가
yline(40, 'r', 'LineWidth',2);

%ylabel('AoI')
legend('Add Std', 'Proposed', 'Threshold');



% X 축 레이블 설정
xticklabels({'EV1', 'EV2', 'EV3', 'EV4', 'EV5', 'EV6', 'EV7', 'EV8','Mean','Std'});

% 그리드 추가
grid on;
