FILE(REMOVE_RECURSE
  "./gpu-module_generated_gpu-module.cu.o"
  "libgpu-module.pdb"
  "libgpu-module.a"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/gpu-module.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
