//
// Created by Benjamin Sylvanus on 3/8/23.
//

#include "controller.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include "simulation.h"
#include "simreader.h"
#include <string.h>
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

controller::controller()
{

}

void controller::Setup(std::string inpath,std::string outpath, int c)
{

  // Request Path?
  this->pathIn = inpath;
  this->pathOut = outpath;
  simreader reader(&this->pathIn);
  printf("Reading Files...\n");
  simulation sim(reader, pathOut);
  this->sim = sim;
  this->view = viewer(c);
  this->commands = {"h","c","a"};

  std::map<std::string, Controls> map{
  {"c", Controls::Commands},
  {"h", Controls::Help},
  {"a", Controls::Args},
  {"s", Controls::Show},
  {"ss",Controls::StepSize},
  {"pp",Controls::PermeationProbability},
  {"d0",Controls::IntrinsicDiffusivity},
  {"ii",Controls::Init_in},
  {"d", Controls::Distance},
  {"ts",Controls::TimeStep},
  {"sc",Controls::Scale},
  {"vs",Controls::VoxelSize},
  {"ns",Controls::NStep},
  {"np",Controls::NPar},
  {"q", Controls::Quit},
  {"Q", Controls::Quit},
  {"done",Controls::Quit},
  {"p", Controls::InPath},
  {"o", Controls::OutPath},
  {"sa", Controls::SaveAll}};

  std::map<std::string, std::string> targets{
  {"c", "Controls::Commands"},
  {"h", "Controls::Help"},
  {"a", "Controls::Args"},
  {"s", "Controls::Show"},
  {"ss","Controls::StepSize"},
  {"pp","Controls::PermeationProbability"},
  {"d0","Controls::IntrinsicDiffusivity"},
  {"ii","Controls::Init_in"},
  {"d", "Controls::Distance"},
  {"ts","Controls::TimeStep"},
  {"sc","Controls::Scale"},
  {"vs","Controls::VoxelSize"},
  {"ns","Controls::NStep"},
  {"np","Controls::NPar"},
  {"q","Controls::Quit"},
  {"Q","Controls::Quit"},
  {"done", "Controls::Quit"},
  {"p", "Controls::InPath"},
  {"o", "Controls::OutPath"},
  {"sa", "Controls::OutPath"}};

  this->map = map;
  this->targets = targets;
  this->args = {"ss", "pp", "d0", "ii", "d", "ts", "pd", "ad", "s", "vs", "ns","np"};
  /**
  *
      step_size
      perm_prob
      D0
      d
      tstep
      parameterdata
      arraydims
      scale
      vsize
  */
}

void controller::Setup(int argc, char ** argv, int c)
{



    std::map<std::string, Controls> map{
            {"c", Controls::Commands},
            {"h", Controls::Help},
            {"a", Controls::Args},
            {"s", Controls::Show},
            {"ss",Controls::StepSize},
            {"pp",Controls::PermeationProbability},
            {"d0",Controls::IntrinsicDiffusivity},
            {"ii",Controls::Init_in},
            {"d", Controls::Distance},
            {"ts",Controls::TimeStep},
            {"sc",Controls::Scale},
            {"vs",Controls::VoxelSize},
            {"ns",Controls::NStep},
            {"np",Controls::NPar},
            {"q", Controls::Quit},
            {"Q", Controls::Quit},
            {"done",Controls::Quit},
            {"p", Controls::InPath},
            {"o", Controls::OutPath},
            {"sa", Controls::SaveAll}};

    std::map<std::string, std::string> targets{
            {"c", "Controls::Commands"},
            {"h", "Controls::Help"},
            {"a", "Controls::Args"},
            {"s", "Controls::Show"},
            {"ss","Controls::StepSize"},
            {"pp","Controls::PermeationProbability"},
            {"d0","Controls::IntrinsicDiffusivity"},
            {"ii","Controls::Init_in"},
            {"d", "Controls::Distance"},
            {"ts","Controls::TimeStep"},
            {"sc","Controls::Scale"},
            {"vs","Controls::VoxelSize"},
            {"ns","Controls::NStep"},
            {"np","Controls::NPar"},
            {"q","Controls::Quit"},
            {"Q","Controls::Quit"},
            {"done", "Controls::Quit"},
            {"p", "Controls::InPath"},
            {"o", "Controls::OutPath"},
            {"sa", "Controls::OutPath"}};

    this->map = map;
    this->targets = targets;
    this->args = {"ss", "pp", "d0", "ii", "d", "ts", "pd", "ad", "s", "vs", "ns","np"};
    this->view = viewer(c);
    this->commands = {"h","c","a"};

    // Parse command line arguments form -arg value pairs
    // construct a string first
    std::string buf;
    for(int i = 0; i < argc; i++)
    {
        buf += argv[i];
        buf += " ";
    }

    // Parse input
    bool b = false;
    this->start(buf,b);
    if (this->pathIn.empty()||this->pathOut.empty())
    {
        printf("Please Configure InPath and OutPath\n");
        exit(0);
    }
}


void controller::handlecommand(std::vector<string> sub, bool * b)
{
    std::string c = sub[0];

    if(this->map.count(c))
    {
    Controls input = this->map[c];
    std::string target = this->targets[c];

    switch(input)
    {
      case Controls::Help:
        view.showHelp();
        break;

      case Controls::Args:
        this->view.showargs(this->targets);
        break;

      case Controls::Commands:
        view.showCommands();
        break;
      case Controls::Show:
        view.show(this->sim);
        break;

      case Controls::NStep:
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setStep_num(std::stod(sub[1]));
        break;

      case Controls::NPar:
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setParticle_num(std::stod(sub[1]));
        break;

      case Controls::StepSize:
        // validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setStep_size(std::stod(sub[1]));
        break;

      case Controls::PermeationProbability:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setPerm_prob(std::stod(sub[1]));
        break;

      case Controls::IntrinsicDiffusivity:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setD0(std::stod(sub[1]));
        break;

      case Controls::Init_in:
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setInit_in(std::stod(sub[1]));
        break;

      case Controls::Distance:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setD(std::stod(sub[1]));
        break;

      case Controls::TimeStep:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setTstep(std::stod(sub[1]));
        break;

      case Controls::Scale:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setScale(std::stod(sub[1]));
        break;

      case Controls::VoxelSize:
        //validate additional parameters;
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setVsize(std::stod(sub[1]));
        break;

      case Controls::Quit:
        *b = true;
        break;

      case Controls::InPath:
        // read a new simulation
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->setSim(sub[1]);
        break;

      case Controls::OutPath:
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->setPathOut(sub[1]);
        break;

      case Controls::SaveAll:
        (sub.size() < 2) ? view.AlertNoParameter(target) : this->sim.setSaveAll(std::stoi(sub[1]));
        break;


      default:
        printf("Invalid Argument\n");
        break;
    }
  }

  else
  {
    printf("Invalid Argument\n");
  }
}

void controller::handleinput(std::string buf, bool * b)
{
  std::vector<std::string> elements = split(buf,"-");

  for(const string& s: elements)
  {
      if (s.length() > 0)
      {
        std::vector<string> sub = split(s," ");
        handlecommand(sub,b);
      }
  }
}

void controller::start(std::string buf, bool b) {
    view.welcome();
    handleinput(buf,&b);
    if (!b) {
      while (true)
      {
          string buff;
          getline(cin,buff);
          handleinput(buff,&b);
          if(b)
          {
             break;
          }
    }
  }
}


void controller::start() {
    view.welcome();
    bool b = false;
    while (true)
    {
        string buf;
        getline(cin,buf);
        handleinput(buf,&b);
        if(b)
        {
           break;
        }
    }


    // calculate Memory required for operation
    // hostBounds = (int *) malloc(3 * sizeof(int));
    // hostdx2 = (double *) malloc(6 * iter * sizeof(double));
    // hostdx4 = (double *) malloc(15 * iter * sizeof(double));
    // hostSimP = (double *) malloc(10 * sizeof(double));
    // hostD4Swc = (double4 *) malloc(nrow * sizeof(double4));
    // hostNewLut = (int *) malloc(prod * sizeof(int));
    // hostNewIndex = (int *) malloc(newindexsize * sizeof(int));
    // hostIndexSize = (int *) malloc(3 * sizeof(int));
    // mdx2 = (double *) malloc(6 * iter * sizeof(double));
    // mdx4 = (double *) malloc(15 * iter * sizeof(double));
    // hostAllData = (double *) malloc(3 * iter * size * sizeof(double));

    // int HOST_MEM_USE;
    // int *
    // 3
    // prod
    // newindexsize
    // 3

    // double *
    // 6 * iter
    // 15 * iter
    // 10
    // 6 * iter
    // 15 * iter
    // 3 * iter * size

    // double4 *
    // nrow

    // hostBounds = (int *) malloc(3 * sizeof(int));
    // hostdx2 = (double *) malloc(6 * iter * sizeof(double));
    // hostdx4 = (double *) malloc(15 * iter * sizeof(double));
    // hostSimP = (double *) malloc(10 * sizeof(double));
    // hostD4Swc = (double4 *) malloc(nrow * sizeof(double4));
    // hostNewLut = (int *) malloc(prod * sizeof(int));
    // hostNewIndex = (int *) malloc(newindexsize * sizeof(int));
    // hostIndexSize = (int *) malloc(3 * sizeof(int));
    // mdx2 = (double *) malloc(6 * iter * sizeof(double));
    // mdx4 = (double *) malloc(15 * iter * sizeof(double));
    // hostAllData = (double *) malloc(3 * iter * size * sizeof(double));



}

simulation controller::getSim()
{
  return this->sim;
}

void controller::setSim(string path)
{
  this->pathIn = path;
  simreader reader(&path);
  printf("Reading Files...\n");
  simulation sim(reader, pathOut);
  this->sim = sim;
}

void controller::setPathOut(string outpath)
{
  this->pathOut = outpath;
  this->sim.setResultPath(outpath);
}
