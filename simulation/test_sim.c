#include "sim.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) check((condition), #condition, __LINE__)

static void check(bool condition, const char *expression, int line)
{
    if (!condition) {
        fprintf(stderr, "test failed at line %d: %s\n", line, expression);
        exit(1);
    }
}

static bool nearly_equal(float first, float second)
{
    return fabsf(first - second) < 0.0001f;
}

static int report_box2d_assert(const char *condition, const char *file, int line)
{
    fprintf(stderr, "Box2D assertion at %s:%d: %s\n", file, line, condition);
    return 1;
}

static void assert_finite_snapshot(SimSnapshot state)
{
    CHECK(isfinite(state.pot_position.x));
    CHECK(isfinite(state.pot_position.y));
    CHECK(isfinite(state.hammer_position.x));
    CHECK(isfinite(state.hammer_position.y));
    CHECK(isfinite(state.extension));
    CHECK(state.extension >= -1.11f);
    CHECK(state.extension <= 0.71f);
}

static void assert_matching_snapshots(SimSnapshot first, SimSnapshot second)
{
    CHECK(first.tick == second.tick);
    CHECK(nearly_equal(first.pot_position.x, second.pot_position.x));
    CHECK(nearly_equal(first.pot_position.y, second.pot_position.y));
    CHECK(nearly_equal(first.hammer_position.x, second.hammer_position.x));
    CHECK(nearly_equal(first.hammer_position.y, second.hammer_position.y));
    CHECK(nearly_equal(first.extension, second.extension));
}

static void test_replay_is_repeatable(void)
{
    Sim *first = sim_create();
    Sim *second = sim_create();
    CHECK(first != NULL && second != NULL);

    for (int i = 0; i < 720; ++i) {
        float dx = (float)((i % 11) - 5) * 0.6f;
        float dy = (float)((i % 7) - 3) * 0.4f;
        sim_step(first, dx, dy);
        sim_step(second, dx, dy);
    }

    SimSnapshot a = sim_snapshot(first);
    SimSnapshot b = sim_snapshot(second);
    CHECK(a.tick == 720 && b.tick == 720);
    assert_matching_snapshots(a, b);
    assert_finite_snapshot(a);
    sim_destroy(first);
    sim_destroy(second);
}

static void test_reset_recreates_the_world(void)
{
    Sim *sim = sim_create();
    CHECK(sim != NULL);
    for (int i = 0; i < 120; ++i) {
        sim_step(sim, 7.0f, -2.0f);
    }
    sim_reset(sim);
    SimSnapshot state = sim_snapshot(sim);
    CHECK(state.tick == 0);
    CHECK(nearly_equal(state.pot_position.x, 0.0f));
    CHECK(nearly_equal(state.pot_position.y, 0.0f));
    assert_finite_snapshot(state);
    sim_destroy(sim);
}

static void test_imported_level(const char *path)
{
    Sim *first = sim_create();
    Sim *second = sim_create();
    CHECK(first != NULL && second != NULL);
    char error[256];
    CHECK(sim_load_level(first, path, error, sizeof(error)));
    CHECK(sim_load_level(second, path, error, sizeof(error)));
    CHECK(strcmp(sim_level_name(first), "Imported collision map") == 0);

    bool touched_terrain = false;
    for (int i = 0; i < 240; ++i) {
        sim_step(first, 0.0f, 0.0f);
        sim_step(second, 0.0f, 0.0f);
        SimSnapshot step = sim_snapshot(first);
        touched_terrain = touched_terrain
            || step.pot_contacts > 0 || step.hammer_contacts > 0;
    }
    SimSnapshot state = sim_snapshot(first);
    assert_matching_snapshots(state, sim_snapshot(second));
    CHECK(state.has_external_level);
    CHECK(state.tick == 240);
    CHECK(state.level_max.y > state.level_min.y);
    CHECK(touched_terrain);
    assert_finite_snapshot(state);

    for (int i = 0; i < 720; ++i) {
        float dx = (float)((i % 13) - 6) * 0.5f;
        float dy = (float)((i % 9) - 4) * 0.4f;
        sim_step(first, dx, dy);
        sim_step(second, dx, dy);
    }
    state = sim_snapshot(first);
    assert_matching_snapshots(state, sim_snapshot(second));
    assert_finite_snapshot(state);

    sim_reset(first);
    state = sim_snapshot(first);
    CHECK(state.has_external_level);
    CHECK(state.tick == 0);
    sim_destroy(first);
    sim_destroy(second);
}

int main(int argc, char **argv)
{
    b2SetAssertFcn(report_box2d_assert);
    test_replay_is_repeatable();
    test_reset_recreates_the_world();
    if (argc == 2) {
        test_imported_level(argv[1]);
    } else {
        CHECK(argc == 1);
    }
    puts("aiget_sim tests passed");
    return 0;
}
