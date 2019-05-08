#include <stdio.h>
#include <string.h>
#include "io.h"


int main(int argc, char** argv){
    if (argc != 4){ // ./see file.txt START LEN
        printf("Specify start and len\n");
        exit(1);
    }
    char* fname = argv[1];
    long long start = atoi(argv[2]);
    int len = atoi(argv[3]);
    char* s = readFile(fname);
    long long l = strlen(s);
    if (start+len > l){
        printf("Larger than tail %lld\n", l);
        exit(1);
    }
    for (long long i = start; i < start+len; i++)
        printf("%c", s[i]);
    printf("\n");

    return 1;
}
