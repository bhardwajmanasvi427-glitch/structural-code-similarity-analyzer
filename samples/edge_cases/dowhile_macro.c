#define LIMIT 10
int f(int n) {
    int t = 0;
    do { t += n; n--; } while (n > 0 && t < LIMIT);
    return t > 5 ? t : 0;
}
