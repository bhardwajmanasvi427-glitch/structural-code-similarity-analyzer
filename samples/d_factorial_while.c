#include <stdio.h>
int factorial(int num) {
    int result = 1;
    int k = 2;
    while (k <= num) {
        result = result * k;
        k++;
    }
    return result;
}
