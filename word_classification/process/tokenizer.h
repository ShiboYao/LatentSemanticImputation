#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "nlt.h"
#include "io.h"

#define MAXLINE 50


struct Token{
    char token[MAXLINE]; //with '_'
    char gram[MAXLINE]; //without '_'
    int n;
};


struct Counter{
    char key[MAXLINE];
    int count;
};


char** readToken(char*, const char*, int*);
char** regToken(char**, const int);
int countGram(char*);
char** filterToken(char**, int, int*);
int gram2token(char*);
struct Token* makeTokenList(char**, int);
int searchWord2(char*, char*, char**, int, int*);
char* moveNgrams(char*, int);
void seize(char*, int, char**, int, struct Counter*);
struct Counter* Tokenize(char*, struct Token*, char**, int);




char** readToken(char* s, const char* header, int* size){
    int col = 0;
    char* p = s;
    
    while (*p != '\n'){ //find out which column is token
        col++;
        char* k = p;
        while (*p != ',' && *p != '\n')
            p++;
        if (wordCompare(k, p, header) == 0){
            while (*p != '\n')
                p++;
            p++;
            break;
        }
        else if (*p == '\n'){
            printf("Header Error\n");
            exit(1);
        }
        else {
            p++;
        }
    }
    s = p;
    int row = 0;
    while (*p){ //token number exluding first header row
        while (*p && *p++ != '\n')
            ;
        row++;
        p++;
    }
    *size = row;
    
    char** result = malloc(sizeof(char*) * row);
    p = s;
    int i = 0;
    while (*p){
        int commas = 1;
        while (commas < col){
            while (*p != ',')
                p++;
            commas++;
            p++;
        }
        result[i] = malloc(sizeof(char)*MAXLINE);
        char* temp = result[i++];
        while (*p && *p != ',' && *p != '\n')
            *temp++ = *p++;
        *temp = '\0';
        while (*p && *p++ != '\n')
            ;
        p++;
    }

    return result;
}


char** regToken(char** token, const int size){
    char** result = malloc(sizeof(char*) * size);

    for (int i = 0; i < size; i++){
        result[i] = malloc(sizeof(char) * (strlen(token[i])+1));
        result[i] = removeSpace(token[i]);
    }

    return result;
}


int countGram(char* gram){
    char* t = gram;
    int count = 0;
    while (*t){
        while (*t && !isspace(*t))
            t++;
        count++;
        if (isspace(*t))
            t++;
    }

    return count;
}


char** filterToken(char** token, int size, int* newSize){
    *newSize = 0;
    for (int i = 0; i < size; i++){
        if (countGram(token[i]) > 1)
            (*newSize)++;
    }
    char** result = malloc(sizeof(char*) * *newSize);
    int j = 0;
    for (int i = 0; i < size; i++){
        if (countGram(token[i]) > 1){
            result[j] = malloc(sizeof(char) * (strlen(token[i])+1));
            lower(token[i]);
            strcpy(result[j], token[i]);
            j++;
        }
    }

    return result;
}


int gram2token(char* gram){ //previuosly token means gram
    char* t = gram;
    int count = 0;
    while (*t){
        while (*t && !isspace(*t))
            t++;
        count++;
        if (isspace(*t))
            *t = '_';
    }

    return count;
}



struct Token* makeTokenList(char** token, int size){
    struct Token* result = malloc(sizeof(struct Token) * size);
    for (int i = 0; i < size; i++){
        strcpy(result[i].token, token[i]);
        strcpy(result[i].gram, token[i]);
        result[i].n = gram2token(result[i].token);
    }

    return result;
}


int searchWord2(char* head, char* end, char** words, int size, int* pos){
    int left = 0; /*in here used as finding ngram that can be tokenized*/
    int right = size-1; /*change size to size-1, Oct 9*/
    int mid = (left+right)/2;

    while (left <= right){
        int r = wordCompare(head, end, words[mid]);
        if (r == 0){
            *pos = mid;
            return 1;
        }
        else if (r > 0){
            left = mid+1;
        }
        else if (r == -1)
            right = mid-1;
        else 
            return 0;
        mid = (left+right)/2;
    }

    return 0;
}


char* moveNgrams(char* p, int n){
    while (n){ //from nonspace, stop at space
        while (*p && *p != ' ')
            p++;
        n--;
        if (*p && n)
            p++;
    }

    return p;
}


void seize(char* s, int n, char** newToken, int size, struct Counter* counter){
    char* p = s; //given gram number, add underlines, count
    char* q = s;
    p = moveNgrams(p, n);
    int pos;
    while (*p){
        if (searchWord2(q, p, newToken, size, &pos) == 1){
            counter[pos].count++;
            //printf("%d  %s\n", counter[pos].count, counter[pos].key);
            char* k = q;
            while (k < p){ //this bug took me an hour to fix!!
                if (*k == ' ')
                    *k = '_';
                k++;
            }  //better finish brackets before write code!!

            p++;
            p = moveNgrams(p, n);
        }
        else {
            p++;
            p = moveNgrams(p, 1);
        }
        q = moveNgrams(q, 1);
        q++;
    }
}


struct Counter* Tokenize(char* s, struct Token* TokenList, char** newToken, int size){
    struct Counter* counter = malloc(sizeof(struct Counter)*size);
    int MAXI = 0, MINI = 999;
    for (int i = 0; i < size; i++){
        strcpy(counter[i].key, newToken[i]);
        counter[i].count = 0;
        if (TokenList[i].n > MAXI)
            MAXI = TokenList[i].n;
        else if (TokenList[i].n < MINI)
            MINI = TokenList[i].n;
    }
    
    for (int i = MAXI; i >= MINI; i--){
        printf("Tokenize %d-gram\n", i);
        seize(s, i, newToken, size, counter);
        printf("%d-gram tokenized\n", i);
    }
    
    return counter;
}


#endif
