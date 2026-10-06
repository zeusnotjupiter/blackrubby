#pragma once

#ifndef BLACKRUBBY_NO_MAIN
#define BLACKRUBBY_NO_MAIN
#define BLACKRUBBY_UNDEFINE_NO_MAIN
#endif

#include "blackrubby_impl.cpp"

#ifdef BLACKRUBBY_UNDEFINE_NO_MAIN
#undef BLACKRUBBY_NO_MAIN
#undef BLACKRUBBY_UNDEFINE_NO_MAIN
#endif