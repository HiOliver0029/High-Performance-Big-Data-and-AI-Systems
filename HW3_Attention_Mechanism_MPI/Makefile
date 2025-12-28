# Compiler settings
CC = gcc
MPICC = mpicc

# Compiler flags
CFLAGS = -O3 -march=native -funroll-loops -Wall -Wextra
LDFLAGS = -lm

# MPI runtime flags (for Docker/root environments)
MPIRUN_FLAGS = --allow-run-as-root

# Target executables
SERIAL_TARGET = attention
MPI_TARGET = attention-mpi

# Source files
SERIAL_SRC = attention.c
MPI_SRC = attention-mpi.c

# Test data files
TEST_FILES = test_tiny.bin test_small.bin test_medium.bin test_large.bin test_xlarge.bin

# Default target
.PHONY: all
all: serial mpi

# Serial version
.PHONY: serial
serial: $(SERIAL_TARGET)

$(SERIAL_TARGET): $(SERIAL_SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "Serial version compiled successfully!"

# MPI version
.PHONY: mpi
mpi: $(MPI_TARGET)

$(MPI_TARGET): $(MPI_SRC)
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS)
	@echo "MPI version compiled successfully!"

# Generate test data
.PHONY: testdata
testdata:
	python test_data_generator.py
	@echo "Test data generated successfully!"

# Run basic tests
.PHONY: test
test: all
	@echo "========================================="
	@echo "Running Basic Tests"
	@echo "========================================="
	@if [ -f test_small.bin ]; then \
		echo "\n--- Testing Serial Version ---"; \
		./$(SERIAL_TARGET) test_small.bin; \
		echo "\n--- Testing MPI Version (4 processes) ---"; \
		mpirun $(MPIRUN_FLAGS) -np 4 ./$(MPI_TARGET) test_small.bin; \
	else \
		echo "Test data not found. Run 'make testdata' first."; \
	fi

# Run comprehensive tests
.PHONY: test-all
test-all: all testdata
	@echo "========================================="
	@echo "Running Comprehensive Tests"
	@echo "========================================="
	@for file in $(TEST_FILES); do \
		if [ -f $$file ]; then \
			echo "\n=== Testing $$file ==="; \
			echo "Serial:"; \
			./$(SERIAL_TARGET) $$file; \
			echo "MPI (4 processes):"; \
			mpirun $(MPIRUN_FLAGS) -np 4 ./$(MPI_TARGET) $$file; \
		fi; \
	done

# Benchmark serial version
.PHONY: bench-serial
bench-serial: serial
	@echo "========================================="
	@echo "Serial Version Benchmark"
	@echo "========================================="
	@if [ -f test_large.bin ]; then \
		echo "\nRunning on test_large.bin..."; \
		./$(SERIAL_TARGET) test_large.bin; \
	fi
	@if [ -f test_xlarge.bin ]; then \
		echo "\nRunning on test_xlarge.bin..."; \
		./$(SERIAL_TARGET) test_xlarge.bin; \
	fi

# Benchmark MPI version with different process counts
.PHONY: bench-mpi
bench-mpi: mpi
	@echo "========================================="
	@echo "MPI Version Benchmark"
	@echo "========================================="
	@if [ -f test_xlarge.bin ]; then \
		for np in 1 2 4 8 16; do \
			echo "\n--- Testing with $$np processes ---"; \
			mpirun $(MPIRUN_FLAGS) -np $$np ./$(MPI_TARGET) test_xlarge.bin; \
		done; \
	else \
		echo "test_xlarge.bin not found. Run 'make testdata' first."; \
	fi

# Clean build artifacts
.PHONY: clean
clean:
	rm -f $(SERIAL_TARGET) $(MPI_TARGET)
	rm -f *.o *.out
	@echo "Build artifacts cleaned!"

# Clean everything including test data
.PHONY: cleanall
cleanall: clean
	rm -f $(TEST_FILES) test_asymmetric.bin test_xxlarge.bin
	@echo "All generated files cleaned!"

# Display help information
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all         - Build both serial and MPI versions (default)"
	@echo "  serial      - Build only serial version"
	@echo "  mpi         - Build only MPI version"
	@echo "  testdata    - Generate test data files"
	@echo "  test        - Run basic tests"
	@echo "  test-all    - Run comprehensive tests on all test files"
	@echo "  bench-serial - Benchmark serial version"
	@echo "  bench-mpi   - Benchmark MPI version with various process counts"
	@echo "  clean       - Remove compiled executables"
	@echo "  cleanall    - Remove executables and test data"
	@echo "  help        - Display this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make              # Build everything"
	@echo "  make testdata     # Generate test data"
	@echo "  make test         # Run basic tests"
	@echo "  make bench-mpi    # Benchmark MPI version"

# Install dependencies (for reference)
.PHONY: install-deps
install-deps:
	@echo "Installing dependencies..."
	@echo "For Ubuntu/Debian:"
	@echo "  sudo apt-get install gcc make libopenmpi-dev openmpi-bin python3 python3-numpy"
	@echo ""
	@echo "For CentOS/RHEL:"
	@echo "  sudo yum install gcc make openmpi openmpi-devel python3 python3-numpy"
	@echo ""
	@echo "For macOS:"
	@echo "  brew install gcc open-mpi python3"
	@echo "  pip3 install numpy"
