#include <stdio.h>
int g(int x) {
    int r = 0;
    do { r += x; x--; } while (x > 0);
    switch (r) { case 1: r = 2; break; default: r = 3; }
    return r > 5 ? r : -r;
}
