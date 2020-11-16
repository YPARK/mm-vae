cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(mmvae
  VERSION 0.1
  DESCRIPTION "")

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

###########################
# Typical Gzipp'ed stream #
###########################

add_library(gzslib STATIC
  src/utils/gzstream.cc
  include/utils/gzstream.hh
  )

target_include_directories(gzslib PUBLIC
  include/utils
  )

####################
# BGZipp'ed stream #
####################

add_library(bgzf STATIC
  src/ext/tabix/bgzf.c
  src/ext/tabix/kstring.c
  include/ext/tabix/bgzf.h
  include/ext/tabix/kstring.h
  )

target_include_directories(bgzf PUBLIC
  include/ext/tabix
  )

add_library(bgzslib STATIC
  src/utils/bgzstream.cc
  include/utils/bgzstream.hh
  )

target_include_directories(bgzslib PUBLIC
  include/utils
  include/ext/tabix
  )

##############
# Executable #
##############

file(GLOB mmvae_SRC
  "include/*.hh"
  "include/utils/*.h"
  "include/utils/*.hh"
  )

add_executable(mmvae mmvae_main.cc mmvae.hh ${mmvae_SRC})
target_include_directories(mmvae PUBLIC include)

target_link_libraries(mmvae "${TORCH_LIBRARIES}")
target_link_libraries(mmvae ZLIB::ZLIB gzslib bgzslib bgzf)


set_property(TARGET mmvae PROPERTY CXX_STANDARD 14)
