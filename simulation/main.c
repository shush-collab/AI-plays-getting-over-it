#include "sim.h"
#include "view.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

enum { max_steps_per_frame = 12 };

typedef struct {
    float seconds;
    float mouse_dx;
    float mouse_dy;
} TickQueue;

static int tick_count(const TickQueue *queue)
{
    int count = (int)(queue->seconds / SIM_TIMESTEP);
    return count < max_steps_per_frame ? count : max_steps_per_frame;
}

static void queue_input(TickQueue *queue, float seconds, GameInput input)
{
    queue->seconds = fminf(queue->seconds + seconds, SIM_TIMESTEP * max_steps_per_frame);
    queue->mouse_dx += input.mouse_dx;
    queue->mouse_dy += input.mouse_dy;
}

static void step_queued_ticks(Sim *sim, TickQueue *queue)
{
    int count = tick_count(queue);
    if (count == 0) {
        return;
    }

    float dx = queue->mouse_dx / (float)count;
    float dy = queue->mouse_dy / (float)count;
    for (int i = 0; i < count; ++i) {
        sim_step(sim, dx, dy);
        queue->seconds -= SIM_TIMESTEP;
    }
    queue->mouse_dx = 0.0f;
    queue->mouse_dy = 0.0f;
}

static void print_usage(const char *program)
{
    fprintf(stderr, "usage: %s [--level collision-map.txt]\n", program);
}

static bool load_requested_level(Sim *sim, int argc, char **argv)
{
    if (argc == 1) {
        return true;
    }
    if (argc != 3 || strcmp(argv[1], "--level") != 0) {
        print_usage(argv[0]);
        return false;
    }

    char error[256];
    if (!sim_load_level(sim, argv[2], error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    Sim *sim = sim_create();
    if (sim == NULL) {
        fputs("Could not create the physics world.\n", stderr);
        return 1;
    }
    if (!load_requested_level(sim, argc, argv)) {
        sim_destroy(sim);
        return 2;
    }

    GameView view = {.zoom = 78.0f, .show_help = true};
    TickQueue queue = {0};
    if (!view_open(&view)) {
        sim_destroy(sim);
        return 1;
    }

    while (!view_should_close()) {
        GameInput input = view_read_input(&view);
        if (input.reset) {
            sim_reset(sim);
            queue = (TickQueue){0};
        }
        queue_input(&queue, view_frame_seconds(), input);
        step_queued_ticks(sim, &queue);
        view_draw(&view, sim);
    }

    view_close();
    sim_destroy(sim);
    return 0;
}
