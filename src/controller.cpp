//
// Created by Benjamin Sylvanus on 3/8/23.
//

#include "controller.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace std;

controller::controller(simulation& sim) : sim(sim) {
    this->view = viewer();
}

void controller::start() {
    view.welcome();
    while (true)
    {
        char buf[100];
        std::cin.getline(buf,100);
        cin.ignore();
        for (char c: buf)
        {
            cout << c;
        }
        cout<<endl;
        break;
    }
}
