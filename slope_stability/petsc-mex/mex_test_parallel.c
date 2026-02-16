#include "mex.h"
#include <petsc.h>
#include <petscmat.h>
#include <petscksp.h>

static PetscBool MyPetscIsInitialized = PETSC_FALSE;

void cleanup(void) {
    PetscBool isInitialized;
    PetscInitialized(&isInitialized);
    if (isInitialized) PetscFinalize();
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2 || !mxIsSparse(prhs[0])) {
        mexErrMsgIdAndTxt("PETSc:Solve:InvalidInput", "Inputs: Sparse Matrix A, Vector b.");
    }

    /* --- 1. INITIALIZATION (Use WORLD for potential parallelism) --- */
    PetscInitialized(&MyPetscIsInitialized);
    if (!MyPetscIsInitialized) {
        PetscInitializeNoArguments();
        /* Disable PETSc signal handling to protect MATLAB */
        PetscOptionsSetValue(NULL, "-no_signal_handler", "true");
        mexAtExit(cleanup);
    }

    mwSize n = mxGetN(prhs[0]);
    double *vals = mxGetPr(prhs[0]);
    mwIndex *ir = mxGetIr(prhs[0]); 
    mwIndex *jc = mxGetJc(prhs[0]); 
    PetscInt nnz = (PetscInt)jc[n];

    /* --- 2. INDEX CONVERSION --- */
    PetscInt *petsc_i, *petsc_j;
    PetscMalloc1(n + 1, &petsc_i);
    PetscMalloc1(nnz, &petsc_j);
    for (mwSize i = 0; i <= n; i++) petsc_i[i] = (PetscInt)jc[i];
    for (mwSize i = 0; i < nnz; i++) petsc_j[i] = (PetscInt)ir[i];

    /* --- 3. MATRIX & VECTOR SETUP --- */
    Mat A;
    /* Use PETSC_COMM_WORLD. On a single MATLAB process, size is 1. */
    MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, (PetscInt)n, (PetscInt)n, petsc_i, petsc_j, vals, &A);

    double *b_ptr = mxGetPr(prhs[1]);
    Vec b, x;
    VecCreateSeqWithArray(PETSC_COMM_WORLD, 1, (PetscInt)n, b_ptr, &b);
    VecDuplicate(b, &x);

    /* --- 4. GMRES SOLVER + GAMG PRECONDITIONER --- */
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);

    /* Set Solver to GMRES */
    KSPSetType(ksp, KSPGMRES);
    KSPGMRESSetRestart(ksp, 30); // Standard restart value

    /* Set Preconditioner to GAMG (Algebraic Multigrid) */
    /* GAMG is much faster than LU/ILU for 10^5+ DOFs in 3D Elasticity */
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCGAMG);

    /* Allow overrides from MATLAB command line (e.g., -ksp_monitor) */
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    /* --- 5. SOLVE --- */
    KSPSolve(ksp, b, x);

    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    
    PetscInt its;
    KSPGetIterationNumber(ksp, &its);

    if (reason > 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Linear solve converged in %d iterations. Reason: %s\n", 
                    (int)its, KSPConvergedReasons[reason]);
    } else {
        PetscPrintf(PETSC_COMM_WORLD, "Linear solve DIVERGED in %d iterations. Reason: %s\n", 
                    (int)its, KSPConvergedReasons[reason]);

    /* --- 6. OUTPUT --- */
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *x_out = mxGetPr(plhs[0]);
    const PetscScalar *x_vals;
    VecGetArrayRead(x, &x_vals);
    for (mwSize i = 0; i < n; i++) x_out[i] = (double)x_vals[i];
    VecRestoreArrayRead(x, &x_vals);

    /* --- 7. CLEANUP --- */
    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);
    PetscFree(petsc_i);
    PetscFree(petsc_j);
}