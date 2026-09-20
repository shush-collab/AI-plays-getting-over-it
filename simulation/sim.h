#ifndef AIGET_SIM_H
#define AIGET_SIM_H

#include <box2d/box2d.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define SIM_TIMESTEP (1.0f / 120.0f)
#define SIM_FINISH_HEIGHT 12.0f

typedef struct Sim Sim;

typedef struct {
    uint64_t tick;
    b2Vec2 pot_position;
    b2Vec2 pot_velocity;
    b2Vec2 hammer_position;
    b2Vec2 hammer_velocity;
    b2Vec2 pivot_position;
    b2Vec2 target_position;
    b2Vec2 level_min;
    b2Vec2 level_max;
    float pot_angle;
    float hammer_angle;
    float extension;
    float max_height;
    float finish_height;
    int hammer_contacts;
    int pot_contacts;
    bool finished;
    bool has_external_level;
} SimSnapshot;

/* Fixed-tick C mechanics. The default course is synthetic; an owned collision
 * map can be loaded separately. It is not yet tick-identical to Unity. */
Sim *sim_create(void);
void sim_destroy(Sim *sim);
void sim_reset(Sim *sim);
/* Transactional: a malformed file leaves the current world unchanged. */
bool sim_load_level(Sim *sim, const char *path, char *error, size_t error_size);
const char *sim_level_name(const Sim *sim);

/* Exactly one fixed tick. Mouse pixels: +x right, +y down. No rendering needed. */
void sim_step(Sim *sim, float mouse_dx, float mouse_dy);
SimSnapshot sim_snapshot(const Sim *sim);
b2WorldId sim_world(const Sim *sim);

#endif
