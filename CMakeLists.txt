cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(mmvae
  VERSION 0.1
  DESCRIPTION "")

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

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

add_executable(nb_vae
  src/nb_vae_main.cc
  include/mmvae.hh
  include/mmvae_io.hh
  include/models/nb.hh
  )

target_include_directories(nb_vae PUBLIC include include/models include/modules)
target_link_libraries(nb_vae "${TORCH_LIBRARIES}")
target_link_libraries(nb_vae ZLIB::ZLIB gzslib bgzslib bgzf)
set_property(TARGET nb_vae PROPERTY CXX_STANDARD 14)

add_executable(vmf_vae
  src/vmf_vae_main.cc
  include/mmvae.hh
  include/operators.hh
  include/mmvae_io.hh
  include/modules/angular.hh
  )

target_include_directories(vmf_vae PUBLIC include include/models include/modules)
target_link_libraries(vmf_vae "${TORCH_LIBRARIES}")
target_link_libraries(vmf_vae ZLIB::ZLIB gzslib bgzslib bgzf)
set_property(TARGET vmf_vae PROPERTY CXX_STANDARD 14)

