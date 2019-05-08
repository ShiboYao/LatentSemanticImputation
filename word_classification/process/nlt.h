#ifndef NLT_H
#define NLT_H


#include "io.h"
#include <string.h>
#include <ctype.h>


char PUNC[] = "!\"$#%\'()*+,-./:;<=>\?@\\^`{|}~_";
/*remove '&' and '['']' from nltk stop*/
/*add escape*/


void removeBracket(char*);
void lower(char*);
char** readWords(const char*, int*);
void sortWords(char**, int);
void printWords(char**, int);
const int wordCompare(char*, char*, const char*);
const int searchWord(char*, char*, char**, const int);
void removeWords(char*, char**, const int);
void swap(char*, char*);
void sortPunc(char*);
const int searchPunc(char, const int);
void removePunc(char*, const int);
char* removeSpace(char*);
void remove_nonASCII(char*);




void removeBraket(char* s){
    char* p = s;

    while (*p){
        if (*p && *p != '['){
            p++;
        }
        else {
            while (*p && *p != ']'){
                *p++ = ' ';
            }
            if (*p)
                *p++ = ' ';
        }
    }
}


void lower(char* s){
    char* p = s;
    int diff = 'a' - 'A';

    while (*p){
        if (*p >= 'A' && *p <= 'Z')
            *p = *p + diff;
        p++;
    }
}


char** readWords(const char* fname, int* size){
    char* s = readFile(fname);
    char* p = s;

    while (*p){
        if (*p == ',')
            (*size)++;
        p++;
    }
    printf("Word list size: %d\n", *size);

    char** result = malloc(sizeof(char*) * (*size));
    int i = 0;
    p = s;

    while (*p){
        char* q = p+1;
        while (*q && *q != ',')
            q++;
        result[i] = malloc(sizeof(char)*(q-p+1));
        char* r = result[i++];
        while (p < q)
            *r++ = *p++;
        *r = '\0';
        p = q+1;
    }

    return result;
}


void sortWords(char** words, const int size){
    char* temp;

    for (int i = 0; i < size; i++){
        for (int j = i + 1; j < size; j++){
            if (strcmp(words[i], words[j]) > 0){
                temp = words[j];
                words[j] = words[i];
                words[i] = temp;
            }
        }
    }
}


void printWords(char** words, int size){
    for (int i = 0; i < size; i++){
        printf("%s ", words[i]);
    }
}


const int wordCompare(char* s, char* end, const char* word){
    char* head = s;
    while (head < end){
        if (*head < *word)
            return -1; /*alphabetally smaller*/
        else if (*head > *word)
            return 1; /*larger*/
        else {
            head++;
            word++;
        }
    }

    if (!*word)
        return 0; /*match*/
    else 
        return -2; /*word is longer than s*/
}


const int searchWord(char* head, char* end, char** words, const int size){
    int left = 0;
    int right = size-1;
    int mid = (left + right) / 2;

    while (left <= right){
        int r = wordCompare(head, end, words[mid]);
        if (r == 0){
            return 1;
        }
        else if (r > 0){
            left = mid + 1;
        } 
        else if (r == -1)
            right = mid - 1;
        else 
            return -2;
        mid = (left + right)/2;
    }

    return 0;
}


void removeWords(char* s, char** words, const int size){
    char* p = s;
    
    while (*p){
        while (isspace(*p))
            p++;
        char* q = p;
        while (*q && !isspace(*q))
            q++;
        if (searchWord(p, q, words, size)){
            while (p < q){
                *p++ = ' ';
            }
        }
        p = q;
    }
}


void swap(char* a, char* b){
    char temp = *a;
    *a = *b;
    *b = temp;
}

void sortPunc(char* Punc){
    for (int i = 0; i < strlen(Punc)-1; i++) {
        for (int j = i+1; j < strlen(Punc); j++) {
            if (Punc[i] > Punc[j]) {
                swap(&Punc[i], &Punc[j]);
            }
        }
    }
}


const int searchPunc(const char c, const int PUNC_LEN){
    int left = 0;
    int right = PUNC_LEN-1;
    int mid = (left + right) / 2;

    while (left <= right){
        if (c == PUNC[mid]){
            return 1;
        }
        else if (c > PUNC[mid]){
            left = mid + 1;
        } 
        else
            right = mid - 1;
        mid = (left + right)/2;
    }

    return 0;
}


void removePunc(char* s, const int PUNC_LEN){
    char* p = s;
    
    while (*p){
        while (*p && isspace(*p))
            p++;
        while (*p && !isspace(*p)){
            if (searchPunc(*p, PUNC_LEN)){
                *p = ' ';
            }
            p++;
        }
    }
}


char* removeSpace(char* s){ 
    char* p = s;
    char* result = malloc(sizeof(char) * (strlen(s)+1));
    char* q = result;
    
    while (*p){
        while (isspace(*p)){
            p++;
        }
        while (*p && !isspace(*p)){
            *q++ = *p++;
        }
        *q++ = ' ';
    }    
    *--q = '\0';

    return result;
}


void remove_nonASCII(char* s){
    char* p = s;
    char* q = s;

    while (*p){
        while (*p && isspace(*p))
            p++;
        q = p;
        while (*p && !isspace(*p) && isascii(*p))
            p++;
        if (!isascii(*p)){
            while (*q && !isspace(*q)){
                *q++ = ' ';
            }
            p = q;
        }
    }
}


#endif
