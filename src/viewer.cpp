//
// Created by Benjamin Sylvanus on 3/8/23.
//

#include "viewer.h"

void viewer::display(int option) {

}

void viewer::show() {

}

void viewer::welcome() {
    printf("---------------\n");
    printf("----Welcome----\n");
    printf("---------------\n");
    printf("%s\n",command.c_str());
}

viewer::viewer()
{
    this->command = "-h help\n-c commands\n -s show configuration -d <arg> show argument\n -<arg> <value> set argument to value\n -args show arguments";

}




