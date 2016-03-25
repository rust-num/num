CARGO_CMD ?= cargo

packages = bigint integer rational traits

test:
	$(MAKE) run-all TASK="test"

run-all: $(packages)
	$(CARGO_CMD) $(TASK)

$(packages):
	$(CARGO_CMD) $(TASK) --manifest-path $@/Cargo.toml

.PHONY: $(packages) test
