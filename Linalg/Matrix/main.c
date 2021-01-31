//
//  main.c
//  Matrix
//
//  Created by Flavian Kaufmann on 31.01.21.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef float matrix_type; // Matrix element type
typedef matrix_type *matrix; // Matrix type
typedef unsigned int idx; // Matrix element index type

void matrix_create(matrix *mat, idx m, idx n); // Creates Matrix of dim mxn
void matrix_create_zeros(matrix *mat, idx m, idx n); // Creates Zero-Matrix of dim mxn
void matrix_create_identity(matrix *mat, idx m, idx n); // Creates Identity-Matrix of dim mxn

void matrix_copy(matrix *dest, matrix *orig, idx m, idx n); // Copies Matrix of dim mxn from orig to dest

void matrix_destroy(matrix *mat); // Destroys Matrix

matrix_type *matrix_element_at_index(matrix *mat, idx m, idx n, idx i, idx j); // Returns elements of matrix of dim mxn at index i,j
void matrix_swap_row(matrix *mat, idx m, idx n, idx row_a, idx row_b); // Swaps 2 rows of Matrix
int matrix_first_nonzero_element_of_column(matrix *mat, idx m, idx n, idx column, idx starting_row, matrix_type tolerance); // Returns index of first nonzero-element in a given column, starting on given row

void matrix_decompose_lup(matrix *mat, matrix* lower, matrix *upper, matrix *p, idx m, idx n); // LR-Decomposition (PA=LU)

void matrix_multiply(matrix *dest, matrix *mat_a, idx m_a, idx n_a, matrix *mat_b, idx m_b, idx n_b); // Matrix Multiplication
void matrix_transpose(matrix *dest, matrix *orig, idx m, idx n); // Transposes Matrix

void matrix_print(matrix *mat, idx m, idx n); // Prints Matrix

int min(int x, int y); // Implementation of min-function


int main(int argc, const char * argv[]) {
    
    matrix mat;
    matrix_create(&mat, 3, 3);
    matrix_type data[] = {1,2,3,
                          2,1,2,
                          3,2,3};
    memcpy(mat, data, sizeof(matrix_type)*9);
    
    printf("Original:\n");
    matrix_print(&mat, 3, 3);
    
    matrix lower;
    matrix upper;
    matrix permutation;
    
    matrix_decompose_lup(&mat, &lower, &upper, &permutation, 3, 3);
    
    printf("\nLower:\n");
    matrix_print(&lower, 3, 3);
    printf("\nUpper:\n");
    matrix_print(&upper, 3, 3);
    printf("\nPermutation:\n");
    matrix_print(&permutation, 3, 3);
    

    
    matrix permutation_transposed;
    matrix_transpose(&permutation_transposed, &permutation, 3, 3);
    
    matrix tmp, result;
    
    matrix_multiply(&tmp, &permutation_transposed, 3, 3, &lower, 3, 3);
    matrix_multiply(&result, &tmp, 3, 3, &upper, 3, 3);
    
    printf("\nResult:\n");
    matrix_print(&result, 3, 3);
    
    matrix_destroy(&mat);
    matrix_destroy(&lower);
    matrix_destroy(&upper);
    matrix_destroy(&permutation);
    matrix_destroy(&permutation_transposed);
    matrix_destroy(&tmp);
    matrix_destroy(&result);
}

// MARK: - Create / Copy / Destroy Matrix

void matrix_create(matrix *mat, idx m, idx n) {
    *mat = calloc(m*n, sizeof(matrix_type));
}

void matrix_create_zeros(matrix *mat, idx m, idx n) {
    matrix_create(mat, m, n);
}

void matrix_create_identity(matrix *mat, idx m, idx n) {
    matrix_create_zeros(mat, m, n);
    idx j = min(m , n);
    for (idx i = 0; i < j; i++) {
        *matrix_element_at_index(mat, m, n, i, i) = 1.0f;
    }
}

void matrix_copy(matrix *dest, matrix *orig, idx m, idx n) {
    matrix_create(dest, m, n);
    memcpy(*dest, *orig, sizeof(matrix_type)*m*n);
}

void matrix_destroy(matrix *mat) {
    free(*mat);
}

// MARK: - Matrix Element Access / Modification

matrix_type *matrix_element_at_index(matrix *mat, idx m, idx n, idx i, idx j) {
    return &((*mat)[n*i + j]);
}

void matrix_swap_row(matrix *mat, idx m, idx n, idx row_a, idx row_b) {
    matrix_type *tmp = malloc(sizeof(matrix_type)*n);
    matrix_type *ptr_row_a = (*mat) + (row_a * n);
    matrix_type *ptr_row_b = (*mat) + (row_b * n);
    memcpy(tmp, ptr_row_a, sizeof(matrix_type)*n);
    memcpy(ptr_row_a, ptr_row_b, sizeof(matrix_type)*n);
    memcpy(ptr_row_b, tmp, sizeof(matrix_type)*n);
    free(tmp);
}

int matrix_first_nonzero_element_of_column(matrix *mat, idx m, idx n, idx column, idx starting_row, matrix_type tolerance) {
    for (int i = starting_row; i < m; i++) {
        if (!(-tolerance < *matrix_element_at_index(mat, m, n, i, column) && *matrix_element_at_index(mat, m, n, i, column) < tolerance)) {
            return i;
        }
    }
    return -1;
}

// MARK: - LU-Decomposition (LR-Zerlegung)

void matrix_decompose_lup(matrix *mat, matrix* lower, matrix *upper, matrix *p, idx m, idx n) {
    matrix_copy(upper, mat, m, n);
    matrix_create_identity(lower, m, n);
    matrix_create_identity(p, m, m);
    
    idx i = 0;
    for (idx j = 0; j < n; j++) {
        if (i >= m) {
            break;
        }
        int index_of_first_nonzero_element = matrix_first_nonzero_element_of_column(upper, m, n, j, i, 0.01f);
        if (index_of_first_nonzero_element == -1) {
            continue;
        }
        
        matrix_swap_row(upper, m, n, i, index_of_first_nonzero_element);
        matrix_swap_row(p, m, m, i, index_of_first_nonzero_element);
        
        for (idx k = i + 1; k < m; k++) {
            matrix_type factor = -(*matrix_element_at_index(upper, m, n, k, j) / *matrix_element_at_index(upper, m, n, i, j));
            *matrix_element_at_index(lower, m, n, k, j) = -factor;
            
            for(idx l = j; l < n; l++) {
                *matrix_element_at_index(upper, m, n, k, l) = *matrix_element_at_index(upper, m, n, k, l) + *matrix_element_at_index(upper, m, n, i, l) * factor;
            }
        }
        i++;
    }
}

// MARK: - Matrix Operations

void matrix_multiply(matrix *dest, matrix *mat_a, idx m_a, idx n_a, matrix *mat_b, idx m_b, idx n_b) {
    matrix_create(dest, m_a, n_b);
    for (idx i = 0; i < m_a; i++) {
        for (idx j = 0; j < n_b; j++) {
            matrix_type *dest_element = matrix_element_at_index(dest, m_a, n_b, i, j);
            for (idx k = 0; k < n_a; k++) {
                *dest_element += (*matrix_element_at_index(mat_a, m_a, n_a, i, k) * *matrix_element_at_index(mat_b, m_b, n_b, k, j));
            }
        }
    }
}

void matrix_transpose(matrix *dest, matrix *orig, idx m, idx n) {
    matrix_create(dest, n, m);
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < n; j++) {
            *matrix_element_at_index(dest, n, m, j, i) = *matrix_element_at_index(orig, m, n, i, j);
        }
    }
}

// MARK: - Debugging

void matrix_print(matrix *mat, idx m, idx n) {
    for (idx i = 0; i < m; i++) {
        for (idx j = 0; j < n; j++) {
            printf("%.2f " , (*mat)[n*i + j]);
        }
        printf("\n");
    }
}

// MARK: - Helpers

int min(int x, int y) {
    return y ^ ((x ^ y) & -(x < y));
}
