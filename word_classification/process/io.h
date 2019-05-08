#ifndef IO_H
#define IO_H

#include <stdio.h>
#include <stdlib.h>


char* readFile(const char* fname){
    FILE* file = fopen(fname, "r");

    fseek(file, 0, SEEK_END);
    long long fsize = ftell(file);
    fseek(file, 0, SEEK_SET); 

    char* s = malloc(sizeof(char)*(fsize + 1));
    fread(s, fsize, 1, file);
    s[fsize] = '\0';
    fclose(file);

    return s;
}


void writeFile(char* fpath, char* s){
    FILE* file = fopen(fpath, "w+");
    if (file != NULL){
        fputs(s, file);
    }
    else 
        exit(1);
    fclose(file);
}


#endif
