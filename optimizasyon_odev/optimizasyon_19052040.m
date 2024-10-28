% Parameters
a = 4;
b = 0;
epsilon = 1e-4;
x0_min = -2;
x0_max = 2;
n = 2; % Dimension of the problem, assuming n = 2

% Define the objective function
f = @(x) 0.5 + (sin(sqrt(x(1)^2 + x(2)^2))^2 - 0.5) / (1 + 0.001 * (x(1)^2 + x(2)^2)^2);
% Define the gradient of the objective function
grad_f = @(x) [
    (2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2)))/(sqrt(x(1)^2 + x(2)^2)*(1000*(x(1)^2 + x(2)^2)^2 + 1));
    (2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2)))/(sqrt(x(1)^2 + x(2)^2)*(1000*(x(1)^2 + x(2)^2)^2 + 1))
];

% Algorithm: Newton-Raphson
x0 = randn(n, 1); % Initial guess from standard normal distribution
x = x0;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -H \ grad_f(x); % Newton-Raphson update
    x = x + delta_x;
end
f_min_newton = f(x);


% Algorithm: Hestenes-Stiefel
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
delta_prev = -grad_prev;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Hestenes-Stiefel update
    x = x + delta_x;
    grad = grad_f(x);
    beta = (grad' * (grad - grad_prev)) / (delta_prev' * (grad - grad_prev)); % Hestenes-Stiefel update
    delta_prev = -grad + beta * delta_prev;
    grad_prev = grad;
end
f_min_hestenes = f(x);


% Algorithm: Polak-Ribière
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
delta_prev = -grad_prev;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Polak-Ribière update
    x = x + delta_x;
    grad = grad_f(x);
    beta = (grad' * (grad - grad_prev)) / (grad_prev' * grad_prev); % Polak-Ribière update
    delta_prev = -grad + beta * delta_prev;
    grad_prev = grad;
end
f_min_polak = f(x);

% Algorithm: Fletcher-Reeves
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
delta_prev = -grad_prev;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Fletcher-Reeves update
    x = x + delta_x;
    grad = grad_f(x);
    beta = (grad' * grad) / (grad_prev' * grad_prev); % Fletcher-Reeves update
    delta_prev = -grad + beta * delta_prev;
    grad_prev = grad;
end
f_min_fletcher = f(x);

% Additional Algorithm: Barzilai-Borwein
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
while norm(grad_f(x)) > epsilon
    grad = grad_f(x);
    if norm(grad - grad_prev) < epsilon
        alpha = 1; % Default alpha is taken as 1 in the first step
    else
        alpha = ((x - x0)' * (grad - grad_prev)) / ((grad - grad_prev)' * (grad - grad_prev)); % Barzilai-Borwein step size
    end
    x0 = x;
    x = x - alpha * grad;
    grad_prev = grad;
end
f_min_barzilai = f(x);

% Display results
fprintf('Minimum of f_ab: Newton-Raphson: %.6f\n', f_min_newton);
fprintf('Minimum of f_ab: Hestenes-Stiefel: %.6f\n', f_min_hestenes);
fprintf('Minimum of f_ab: Polak-Ribière: %.6f\n', f_min_polak);
fprintf('Minimum of f_ab: Fletcher-Reeves: %.6f\n', f_min_fletcher);
fprintf('Minimum of f_ab: Barzilai-Borwein: %.6f\n', f_min_barzilai);
algorithms = {'Newton-Raphson', 'Hestenes-Stiefel', 'Polak-Ribière', 'Fletcher-Reeves', 'Barzilai-Borwein'};
f_min = [f_min_newton, f_min_hestenes, f_min_polak, f_min_fletcher, f_min_barzilai];

algorithms = {'Newton-Raphson', 'Hestenes-Stiefel', 'Polak-Ribière', 'Fletcher-Reeves', 'Barzilai-Borwein'};
desired_iterations = [4, 9, 8, 10, 6]; % İstenen iterasyon sayıları

fprintf('Benchmark Table:\n');
fprintf('-------------------------------------------------------------\n');
fprintf('| Algorithm          |   k   |      x1     |      x2     |   f(x)    | Abs. Error |\n');
fprintf('-------------------------------------------------------------\n');

algorithms = {'Newton-Raphson', 'Hestenes-Stiefel', 'Polak-Ribière', 'Fletcher-Reeves', 'Barzilai-Borwein'};
desired_iterations = [4, 9, 8, 10, 6]; % Belirli algoritmalara karşılık gelen istenen iterasyon sayıları

for alg_idx = 1:length(algorithms)
    algorithm = algorithms{alg_idx};
    max_iterations = desired_iterations(alg_idx);
    x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Başlangıç tahmini uniform dağılımdan alınır
    x = x0;
    grad_prev = grad_f(x);
    delta_prev = -grad_prev;
    iteration = 0;
    while (norm(grad_f(x)) > epsilon) && (iteration < max_iterations)
        iteration = iteration + 1;
        H = [3*x(1)^2 - 1, 0; 0, 1]; % Hesse matrisi
        if strcmp(algorithm, 'Newton-Raphson')
            delta_x = -H \ grad_f(x); % Newton-Raphson güncellemesi
        elseif strcmp(algorithm, 'Hestenes-Stiefel')
            delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Hestenes-Stiefel güncellemesi
        elseif strcmp(algorithm, 'Polak-Ribière')
            delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Polak-Ribière güncellemesi
        elseif strcmp(algorithm, 'Fletcher-Reeves')
            delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Fletcher-Reeves güncellemesi
        elseif strcmp(algorithm, 'Barzilai-Borwein')
            grad = grad_f(x);
            if norm(grad - grad_prev) < epsilon
                alpha = 1; % Default alpha ilk adımda 1 olarak alınır
            else
                alpha = ((x - x0)' * (grad - grad_prev)) / ((grad - grad_prev)' * (grad - grad_prev)); % Barzilai-Borwein adım boyu
            end
            x0 = x;
            x = x - alpha * grad;
            grad_prev = grad;
            fx = f(x);
            abs_error = abs(fx - f_min(alg_idx));
            fprintf('| %s | %d | %.6f | %.6f | %.6f | %.6f |\n', algorithm, iteration, x(1), x(2), fx, abs_error);
            continue;
        end
        x = x + delta_x;
        grad = grad_f(x);
        if strcmp(algorithm, 'Hestenes-Stiefel') || strcmp(algorithm, 'Polak-Ribière') || strcmp(algorithm, 'Fletcher-Reeves')
            beta = (grad' * (grad - grad_prev)) / (delta_prev' * (grad - grad_prev)); % Hestenes-Stiefel, Polak-Ribière, Fletcher-Reeves için beta güncellemesi
            delta_prev = -grad + beta * delta_prev;
        end
        grad_prev = grad;
        fx = f(x);
        abs_error = abs(fx - f_min(alg_idx));
        fprintf('| %s | %d | %.6f | %.6f | %.6f | %.6f |\n', algorithm, iteration, x(1), x(2), fx, abs_error);
    end
end

fprintf('-------------------------------------------------------------\n');


% Define range for x1 and x2
x1_range = linspace(-2, 2, 100);
x2_range = linspace(-2, 2, 100);

% Create meshgrid for x1 and x2
[X1, X2] = meshgrid(x1_range, x2_range);

% Define optimization algorithms
algorithms = {'Hestenes-Stiefel', 'Polak-Ribière', 'Fletcher-Reeves', 'Barzilai-Borwein'};

% Loop over each algorithm
for algo_idx = 1:numel(algorithms)
    algorithm = algorithms{algo_idx};

    % Compute function values for each pair of x1 and x2
    Z = zeros(size(X1));
    for i = 1:numel(X1)
        Z(i) = f([X1(i); X2(i)]);
    end

    % Plot the mesh
    figure;
    mesh(X1, X2, Z);
    xlabel('x1');
    ylabel('x2');
    zlabel('f(x)');
    title(['Mesh Plot of Objective Function (' algorithm ' Algorithm)']);
end



% Algorithm: Newton-Raphson
tic; % Start time
x0 = randn(n, 1); % Initial guess from standard normal distribution
x = x0;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -H \ grad_f(x); % Newton-Raphson update
    x = x + delta_x;
end
f_min_newton = f(x);
elapsed_time_newton = toc; % Elapsed time

fprintf('Elapsed time for Newton-Raphson: %.6f seconds\n', elapsed_time_newton);

% Algorithm: Hestenes-Stiefel
tic; % Start time
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
delta_prev = -grad_prev;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Hestenes-Stiefel update
    x = x + delta_x;
    grad = grad_f(x);
    beta = (grad' * (grad - grad_prev)) / (delta_prev' * (grad - grad_prev)); % Hestenes-Stiefel update
    delta_prev = -grad + beta * delta_prev;
    grad_prev = grad;
end
f_min_hestenes = f(x);
elapsed_time_hestenes = toc; % Elapsed time

fprintf('Elapsed time for Hestenes-Stiefel: %.6f seconds\n', elapsed_time_hestenes);

% Algorithm: Polak-Ribière
tic; % Start time
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
delta_prev = -grad_prev;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Polak-Ribière update
    x = x + delta_x;
    grad = grad_f(x);
    beta = (grad' * (grad - grad_prev)) / (grad_prev' * grad_prev); % Polak-Ribière update
    delta_prev = -grad + beta * delta_prev;
    grad_prev = grad;
end
f_min_polak = f(x);
elapsed_time_polak = toc; % Elapsed time

fprintf('Elapsed time for Polak-Ribière: %.6f seconds\n', elapsed_time_polak);

% Algorithm: Fletcher-Reeves
tic; % Start time
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
delta_prev = -grad_prev;
while norm(grad_f(x)) > epsilon
    H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
         2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
    delta_x = -delta_prev' * grad_prev / (delta_prev' * H * delta_prev) * delta_prev; % Fletcher-Reeves update
    x = x + delta_x;
    grad = grad_f(x); % Compute the gradient
end


% Additional Algorithm: Barzilai-Borwein
tic; % Start time
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; %

% Additional Algorithm: Barzilai-Borwein
tic; % Başlangıç zamanını kaydet
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
while norm(grad_f(x)) > epsilon
    grad = grad_f(x);
    if norm(grad - grad_prev) < epsilon
        alpha = 1; % Default alpha is taken as 1 in the first step
    else
        H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
             2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
        alpha = ((x - x0)' * (grad - grad_prev)) / ((grad - grad_prev)' * (grad - grad_prev)); % Barzilai-Borwein step size
    end
    x0 = x;
    x = x - alpha * grad;
    grad_prev = grad;
end
f_min_barzilai = f(x);
elapsed_time_barzilai = toc; % Geçen zamanı al

fprintf('Elapsed time for Barzilai-Borwein: %.6f seconds\n', elapsed_time_barzilai);
   
% Additional Algorithm: Barzilai-Borwein
tic; % Start time
x0 = rand(n, 1) * (x0_max - x0_min) + x0_min; % Initial guess from uniform distribution
x = x0;
grad_prev = grad_f(x);
while norm(grad_f(x)) > epsilon
    grad = grad_f(x);
    if norm(grad - grad_prev) < epsilon
        alpha = 1; % Default alpha is taken as 1 in the first step
    else
        H = [2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2);...
             2*x(2)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2), 2*x(1)*cos(sqrt(x(1)^2 + x(2)^2))*sin(sqrt(x(1)^2 + x(2)^2))/sqrt(x(1)^2 + x(2)^2)]; % Hessian matrix
        alpha = ((x - x0)' * (grad - grad_prev)) / ((grad - grad_prev)' * (grad - grad_prev)); % Barzilai-Borwein step size
    end
    x0 = x;
    x = x - alpha * grad;
    grad_prev = grad;
end
f_min_barzilai = f(x);
elapsed_time_barzilai = toc; % Get elapsed time

fprintf('Elapsed time for Barzilai-Borwein: %.6f seconds\n', elapsed_time_barzilai);
