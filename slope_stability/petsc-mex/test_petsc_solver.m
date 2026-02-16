n = 1e4; 
fprintf('Problem size: n = %d\n', n);

e = ones(n, 1);
A = spdiags([-e, 2*e, -e], [-1, 0, 1], n, n);

% Let's solve -u'' = f, where f = 1
b = ones(n, 1) * (1/n^2); 

% Boundary Conditions (Dirichlet)
% We set u(1) = 0 and u(n) = 0
A(1, :) = 0; A(1, 1) = 1; b(1) = 0;
A(n, :) = 0; A(n, n) = 1; b(n) = 0;

% Solve using PETSc MEX
fprintf('Solving with PETSc (LU)...\n');
tic;
% x = mex_test(A, b);
% x = mex_test_parallel(A, b);
x = mex_parallel_solve(A, b, '-pc_type', 'hypre', '-pc_hypre_type', 'boomeramg', '-ksp_monitor');
t_solve = toc;
fprintf('Solve completed in %.4f seconds.\n', t_solve);

% Verify Results
residual = norm(A*x - b);
fprintf('Residual Norm: %e\n', residual);

% Plot result (should be a parabola)
plot(linspace(0, 1, n), x, 'LineWidth', 2);
grid on;
title(sprintf('Solution to 1D Laplacian (n=%d)', n));
xlabel('x'); ylabel('u(x)');