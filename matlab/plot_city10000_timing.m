hybrid_smoother_update_times = dlmread('../hybrid_performance/hybrid_timing/hybrid_timing.txt');
isam2_update_times = dlmread('../hybrid_performance/hybrid_timing/ISAM2_timing.txt');
mh_isam2_update_times = dlmread('../hybrid_performance/hybrid_timing/MH_ISAM2_timing.txt');

figure('position', [500, 500, 500, 200]);
hold on;
lineWidth = 1.5;
semilogy(isam2_update_times(:, 1), isam2_update_times(:, 2), '-', 'LineWidth', lineWidth, 'color', [0.1 0.1 0.9]);
semilogy(hybrid_smoother_update_times(:, 1), hybrid_smoother_update_times(:, 2), '-', 'LineWidth', lineWidth, 'color', [0.1 0.9 0.1]);
semilogy(mh_isam2_update_times(:, 1), mh_isam2_update_times(:, 2), '-', 'LineWidth', lineWidth, 'color', [0.7 0.1 0.1]);
grid on;
xlabel('Number of Timesteps');
ylabel('Cumulative time (secs) log scale');
legend('iSAM2', 'Hybrid Factor Graphs', 'MH-iSAM2');
% title('Time taken for each smoother update step');
hold off;
