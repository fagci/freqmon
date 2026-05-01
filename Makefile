CXX      := g++
CXXFLAGS := -O3 -march=native -ffast-math -std=c++23 -fpermissive
RTLSDR   := /usr/local/lib/librtlsdr/local

# Static link librtlsdr (with RF acceleration), dynamic SDL2/ttf/liquid
freqmon: main.cpp ui.h
	$(CXX) $(CXXFLAGS) -I$(RTLSDR)/include \
		-o $@ main.cpp \
		$(RTLSDR)/lib/librtlsdr.a \
		-lSDL2 -lSDL2_ttf -lliquid -lpthread -lusb-1.0

run: freqmon
	./freqmon

clean:
	rm -f freqmon freqmon-static

.PHONY: run clean
