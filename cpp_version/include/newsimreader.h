struct Variable {
    std::string name;
    std::string type;
    std::vector<int> size;
    std::variant<double*, int64_t*> data;
    int order;  // Add this line
};