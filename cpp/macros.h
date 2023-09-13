// macro fuckery

#define DISPATCH_SWITCH(TYPE, NAME, ...) \
  [&] {                                  \
    const auto& _type = TYPE;            \
    const auto& _name = NAME;            \
    switch (_type) {                     \
      __VA_ARGS__                        \
      default:                           \
        AT_ERROR(                        \
          _name,                         \
          ": invalid type `",            \
          toString(_type),               \
          "`"                            \
        );                               \
    }                                    \
  }()                                    \

#define BITWIDTH_CASE(bitwidth, ...)     \
  case bitwidth: {                       \
    unsigned int bit_width = bitwidth;   \
    return __VA_ARGS__();                \
  }

#define BITWIDTH_TYPES(...)              \
  BITWIDTH_CASE(8, __VA_ARGS__)          \
  BITWIDTH_CASE(4, __VA_ARGS__)          \
  BITWIDTH_CASE(2, __VA_ARGS__)          \
  BITWIDTH_CASE(1, __VA_ARGS__)          \

#define DISPATCH_BITWIDTH(TYPE, ...)     \
  DISPATCH_SWITCH(                       \
    TYPE,                                \
    "bitwidth",                          \
    BITWIDTH_TYPES(__VA_ARGS__)          \
  )                                      \

