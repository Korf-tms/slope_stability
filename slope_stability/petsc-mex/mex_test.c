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

    /* --- 1. SAFE INITIALIZATION --- */
    PetscInitialized(&MyPetscIsInitialized);
    if (!MyPetscIsInitialized) {
        /* Disable PETSc signal handling so it doesn't crash MATLAB on errors */
        PetscOptionsSetValue(NULL, "-no_signal_handler", "true");
        PetscInitializeNoArguments();
        mexAtExit(cleanup);
    }

    /* --- 2. DATA EXTRACTION --- */
    mwSize m = mxGetM(prhs[0]); // Rows
    mwSize n = mxGetN(prhs[0]); // Cols
    double *vals = mxGetPr(prhs[0]);
    mwIndex *ir = mxGetIr(prhs[0]); // Row indices (CSC)
    mwIndex *jc = mxGetJc(prhs[0]); // Col pointers (CSC)
    PetscInt nnz = (PetscInt)jc[n];

    /* --- 3. INDEX SIZE CONVERSION --- */
    /* We MUST copy indices because mwIndex (64-bit) != PetscInt (usually 32-bit) */
    PetscInt *petsc_i, *petsc_j;
    PetscMalloc1(n + 1, &petsc_i);
    PetscMalloc1(nnz, &petsc_j);

    for (mwSize i = 0; i <= n; i++) petsc_i[i] = (PetscInt)jc[i];
    for (mwSize i = 0; i < nnz; i++) petsc_j[i] = (PetscInt)ir[i];

    /* --- 4. MATRIX WRAPPING --- */
    Mat A;
    /* Since MATLAB is CSC and PETSc is CSR:
       Passing MATLAB's CSC to PETSc's CSR creates A^T.
       For symmetric FEM elasticity, A = A^T, so this is fine. */
    MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, (PetscInt)m, (PetscInt)n, petsc_i, petsc_j, vals, &A);

    /* --- 5. SOLVER SETUP --- */
    double *b_ptr = mxGetPr(prhs[1]);
    Vec b, x;
    VecCreateSeqWithArray(PETSC_COMM_SELF, 1, (PetscInt)m, b_ptr, &b);
    VecDuplicate(b, &x);

    KSP ksp;
    KSPCreate(PETSC_COMM_SELF, &ksp);
    KSPSetOperators(ksp, A, A);
    KSPSetType(ksp, KSPPREONLY);
    
    PC pc;
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);
    
    /* Crucial for LU: some solvers require this call to initialize defaults */
    KSPSetFromOptions(ksp);
    KSPSetUp(ksp);

    /* --- 6. SOLVE --- */
    PetscErrorCode ierr = KSPSolve(ksp, b, x);
    if (ierr) mexErrMsgTxt("PETSc KSPSolve failed.");

    /* --- 7. OUTPUT --- */
    plhs[0] = mxCreateDoubleMatrix(m, 1, mxREAL);
    double *x_out = mxGetPr(plhs[0]);
    const PetscScalar *x_vals;
    VecGetArrayRead(x, &x_vals);
    for (mwSize i = 0; i < m; i++) x_out[i] = (double)x_vals[i];
    VecRestoreArrayRead(x, &x_vals);

    /* --- 8. CLEANUP --- */
    KSPDestroy(&ksp);
    VecDestroy(&b);
    VecDestroy(&x);
    MatDestroy(&A);
    PetscFree(petsc_i);
    PetscFree(petsc_j);
}