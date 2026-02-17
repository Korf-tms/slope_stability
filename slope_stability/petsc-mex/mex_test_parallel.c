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
    PetscErrorCode ierr;

    /* --- 1. INITIALIZATION --- */
    PetscInitialized(&MyPetscIsInitialized);
    if (!MyPetscIsInitialized) {
        PetscInitializeNoArguments();
        PetscOptionsSetValue(NULL, "-no_signal_handler", "true");
        mexAtExit(cleanup);
    }

    /* Check minimum inputs */
    if (nrhs < 2 || !mxIsSparse(prhs[0])) {
        mexErrMsgTxt("Usage: x = mex_solve(A, b, [block_size], [options_string])");
    }

    /* --- 2. PARSE OPTIONAL ARGUMENTS --- */
    PetscInt bs = 1;
    if (nrhs >= 3 && !mxIsEmpty(prhs[2])) {
        bs = (PetscInt)mxGetScalar(prhs[2]);
    }

    if (nrhs >= 4 && mxIsChar(prhs[3])) {
        char *options_buffer = mxArrayToString(prhs[3]);
        PetscOptionsInsertString(NULL, options_buffer);
        mxFree(options_buffer);
    }

    /* --- 3. DATA EXTRACTION --- */
    mwSize n = mxGetN(prhs[0]);
    double *vals = mxGetPr(prhs[0]);
    mwIndex *ir = mxGetIr(prhs[0]);
    mwIndex *jc = mxGetJc(prhs[0]);
    PetscInt nnz = (PetscInt)jc[n];

    PetscInt *petsc_i, *petsc_j;
    PetscMalloc1(n + 1, &petsc_i);
    PetscMalloc1(nnz, &petsc_j);
    for (mwSize i = 0; i <= n; i++) petsc_i[i] = (PetscInt)jc[i];
    for (mwSize i = 0; i < nnz; i++) petsc_j[i] = (PetscInt)ir[i];

    /* --- 4. PETSc OBJECT SETUP --- */
    Mat A;
    MatCreateSeqAIJWithArrays(PETSC_COMM_WORLD, (PetscInt)n, (PetscInt)n, petsc_i, petsc_j, vals, &A);
    MatSetBlockSize(A, bs);

    double *b_ptr = mxGetPr(prhs[1]);
    Vec b, x;
    VecCreateSeqWithArray(PETSC_COMM_WORLD, 1, (PetscInt)n, b_ptr, &b);
    VecDuplicate(b, &x);

    /* --- 5. SOLVER SETUP --- */
    KSP ksp;
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    /* --- 6. SOLVE --- */
    KSPSolve(ksp, b, x);

    /* --- 7. OUTPUT & CONVERGENCE INFO --- */
    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);
    if (reason < 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Solve DIVERGED with reason %d\n", (int)reason);
    } else {
        PetscInt its;
        KSPGetIterationNumber(ksp, &its);
        PetscPrintf(PETSC_COMM_WORLD, "Solve CONVERGED in %d iterations\n", (int)its);
    }

    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    double *x_out = mxGetPr(plhs[0]);
    const PetscScalar *x_vals;
    VecGetArrayRead(x, &x_vals);
    for (int i = 0; i < n; i++) x_out[i] = (double)x_vals[i];
    VecRestoreArrayRead(x, &x_vals);

    /* --- 8. CLEANUP --- */
    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);
    PetscFree(petsc_i);
    PetscFree(petsc_j);
}
