#ifndef AIGET_VIEW_H
#define AIGET_VIEW_H

#include "sim.h"

#include <stdbool.h>

typedef struct {
    float zoom;
    bool show_help;
} GameView;

typedef struct {
    float mouse_dx;
    float mouse_dy;
    bool reset;
} GameInput;

bool view_open(GameView *view);
void view_close(void);
bool view_should_close(void);
float view_frame_seconds(void);
GameInput view_read_input(GameView *view);
void view_draw(GameView *view, const Sim *sim);

#endif
