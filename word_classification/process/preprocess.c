#include "nlt.h"
#include "tokenizer.h"


int main(int argc, char** argv){
    if (argc != 2){
        printf("Specify raw fname\n");
        exit(1);
    }
    char* s = readFile(argv[1]);
    long len = strlen(s);
    printf("Total char count: %ld\n", len);

    /*do not recommend swtich any of the processes below*/

    int PUNC_LEN = strlen(PUNC);
    sortPunc(PUNC); //need to sort the punctuations in ASCII order

    removeBraket(s); //remove [edit]
    printf("[edit] and [referece] removed.\n");
    removePunc(s, PUNC_LEN); //remove punctuations
    printf("Punctuations removed.\n");
    remove_nonASCII(s);
    printf("Words containing nonASCII removed.\n");
    lower(s); //convert to lower case
    printf("Converted to lower case.\n");
    s = removeSpace(s);
    printf("Extra white spaces removed.\n");

    char* s_token = readFile("sp500_token.csv");
    int Tsize = 0;
    char** token = readToken(s_token, "Token", &Tsize);
    token = regToken(token, Tsize);
    printf("Token extra space checked.\n");

    for (int i = 0; i < Tsize; i++)
        printf("%s\n", token[i]);
    
    int newSize;
    char** newToken = filterToken(token, Tsize, &newSize);
    printf("1-gram removed.\n");
    sortWords(newToken, newSize);
    printf("Token array sorted in alphabetic order.\n");
    struct Token* TokenList = makeTokenList(newToken, newSize);
    printf("TokenList constructed.\n");


    struct Counter* counter = Tokenize(s, TokenList, newToken, newSize);

    for (int i = 0; i < newSize; i++){
        printf("%d  %s\n", counter[i].count, counter[i].key);
    }


    int size = 0; //need to tokenize first then remove stopwords
    char** words = readWords("stopwords.txt", &size);
    sortWords(words, size); //sort the stopwords in alphabeta order
    removeWords(s, words, size); //remove stopwords from file
    printf("Stopwords removed.\n");
    s = removeSpace(s);
    printf("Extra space removed\n again.\n");

    char* fpath = "processed.txt";
    writeFile(fpath, s);
    printf("Saved as processed.txt\n");

    return 1;
}


