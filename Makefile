# Build the SIPNET binary that pySIPNET drives.
#
# SIPNET lives in the sipnet/ git submodule, pinned to a specific release tag.
# From SIPNET v2.0.0 onward every model option (snow, litter pool, nitrogen
# cycle, and so on) is chosen at run time in the sipnet.in config file, so
# there is exactly one binary to build and no compiler flags to pass. Earlier
# SIPNET versions chose those options at compile time, which is why older
# versions of this Makefile built one binary per option combination.

SIPNET_DIR := sipnet
CACHE_DIR  := .sipnet_cache
BINARY     := $(CACHE_DIR)/sipnet

.PHONY: sipnet submodule clean-sipnet

# Default target: build SIPNET and copy the binary into the cache directory
# where pysipnet.runner looks for it.
sipnet: submodule
	$(MAKE) -C $(SIPNET_DIR) clean
	$(MAKE) -C $(SIPNET_DIR)
	mkdir -p $(CACHE_DIR)
	cp $(SIPNET_DIR)/sipnet $(BINARY)
	@echo "Built: $(BINARY)"
	@$(BINARY) --version

# Fetch the submodule contents if this is a fresh clone.
submodule:
	@test -f $(SIPNET_DIR)/Makefile || git submodule update --init $(SIPNET_DIR)

clean-sipnet:
	$(MAKE) -C $(SIPNET_DIR) clean
	rm -rf $(CACHE_DIR)
