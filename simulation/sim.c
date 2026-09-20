#include "sim.h"

#include <math.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Controller gains are provisional. Shapes, masses, joint limits, and motor
 * limits below come from the owned Unity scene. */
static const float mouse_scale = 0.012f;
static const float slider_lower_translation = -1.10f;
static const float slider_upper_translation = 0.70f;
static const float handle_to_pole = 1.172f;
static const float pole_to_tip = 1.080f;
static const float tip_reach_offset = 2.252f;
static const float initial_slider_translation = -0.357f;
static const float initial_slider_angle = -0.638f;
static const b2Vec2 player_hub_anchor = {0.034f, 0.353f};

typedef struct {
    b2Vec2 *points;
    int count;
    b2Vec2 center;
    float radius;
    float friction;
    float restitution;
} LevelShape;

typedef struct {
    LevelShape *shapes;
    size_t count;
    b2Vec2 spawn;
    b2Vec2 min;
    b2Vec2 max;
} Level;

struct Sim {
    b2WorldId world;
    b2BodyId pot;
    b2BodyId hub;
    b2BodyId slider_body;
    b2BodyId handle;
    b2BodyId pole;
    b2BodyId tip;
    b2JointId pot_hub;
    b2JointId hub_slider;
    b2JointId slider_handle;
    b2JointId handle_pole;
    b2JointId pole_tip;
    b2Vec2 target;
    float max_height;
    uint64_t tick;
    Level level;
};

static float clamp(float value, float low, float high)
{
    return fminf(fmaxf(value, low), high);
}

static float angle_error(float target, float actual)
{
    return atan2f(sinf(target - actual), cosf(target - actual));
}

static b2ShapeDef shape_definition(float density, float friction, uint32_t color)
{
    b2ShapeDef shape = b2DefaultShapeDef();
    shape.density = density;
    shape.material.friction = friction;
    shape.material.customColor = color;
    shape.filter.groupIndex = density > 0.0f ? -1 : 0;
    return shape;
}

static b2BodyId create_static_body(Sim *sim, b2Vec2 position, float angle)
{
    b2BodyDef body = b2DefaultBodyDef();
    body.type = b2_staticBody;
    body.position = position;
    body.rotation = b2MakeRot(angle);
    return b2CreateBody(sim->world, &body);
}

static b2BodyId create_dynamic_body(
    Sim *sim,
    b2Vec2 position,
    float angle,
    float linear_damping,
    float angular_damping,
    bool fixed_rotation,
    bool bullet)
{
    b2BodyDef body = b2DefaultBodyDef();
    body.type = b2_dynamicBody;
    body.position = position;
    body.rotation = b2MakeRot(angle);
    body.linearDamping = linear_damping;
    body.angularDamping = angular_damping;
    body.fixedRotation = fixed_rotation;
    body.isBullet = bullet;
    return b2CreateBody(sim->world, &body);
}

static void set_body_mass(b2BodyId body, float mass)
{
    b2MassData data = b2Body_GetMassData(body);
    if (data.mass > 0.0f) {
        float scale = mass / data.mass;
        data.mass = mass;
        data.rotationalInertia *= scale;
    } else {
        data.mass = mass;
        data.center = b2Vec2_zero;
        data.rotationalInertia = mass * 0.10f;
    }
    b2Body_SetMassData(body, data);
}

static void add_platform(Sim *sim, float x, float y, float width, float height, float angle)
{
    b2BodyId body = create_static_body(sim, (b2Vec2){x, y}, angle);
    b2ShapeDef shape = shape_definition(0.0f, 0.85f, 0x788A92);
    b2Polygon polygon = b2MakeBox(width / 2.0f, height / 2.0f);
    b2CreatePolygonShape(body, &shape, &polygon);
}

static void create_course(Sim *sim)
{
    /* Purpose-built practice course, NOT the original game's map. */
    static const float platforms[][5] = {
        {-2.0f, -0.50f, 20.0f, 1.00f,  0.00f},
        { 2.0f,  0.35f,  1.3f, 0.70f, -0.10f},
        { 3.5f,  1.05f,  1.6f, 2.10f,  0.00f},
        { 2.0f,  2.65f,  2.4f, 0.30f,  0.00f},
        {-0.5f,  3.60f,  0.9f, 2.20f,  0.00f},
        { 1.1f,  5.05f,  2.3f, 0.30f,  0.00f},
        { 3.5f,  5.90f,  1.5f, 0.35f,  0.12f},
        { 4.0f,  6.55f,  0.2f, 1.00f,  0.00f},
        { 2.8f,  7.45f,  0.6f, 0.20f,  0.00f},
        { 1.0f,  8.35f,  1.3f, 0.30f, -0.12f},
        {-1.0f,  9.50f,  0.7f, 0.25f,  0.00f},
        { 0.7f, 10.55f,  1.0f, 0.30f,  0.00f},
        { 2.8f, 11.55f,  2.0f, 0.40f,  0.00f},
    };
    for (size_t i = 0; i < sizeof(platforms) / sizeof(platforms[0]); ++i) {
        const float *p = platforms[i];
        add_platform(sim, p[0], p[1], p[2], p[3], p[4]);
    }
}

static void create_level(Sim *sim)
{
    if (sim->level.count == 0) {
        create_course(sim);
        return;
    }
    b2BodyId body = create_static_body(sim, b2Vec2_zero, 0.0f);
    for (size_t i = 0; i < sim->level.count; ++i) {
        const LevelShape *source = &sim->level.shapes[i];
        b2ShapeDef shape = shape_definition(0.0f, source->friction, 0x788A92);
        shape.material.restitution = source->restitution;
        if (source->radius > 0.0f) {
            b2Circle circle = {.center = source->center, .radius = source->radius};
            b2CreateCircleShape(body, &shape, &circle);
        } else if (source->count == 3) {
            b2Hull hull = b2ComputeHull(source->points, 3);
            b2Polygon polygon = b2MakePolygon(&hull, 0.0f);
            b2CreatePolygonShape(body, &shape, &polygon);
        } else {
            /* Exact outline, but one-sided chains are NOT Unity's solid decomposition. */
            b2ChainDef chain = b2DefaultChainDef();
            chain.points = source->points;
            chain.count = source->count;
            chain.materials = &shape.material;
            chain.materialCount = 1;
            chain.isLoop = true;
            b2CreateChain(body, &chain);
        }
    }
}

static void create_pot(Sim *sim)
{
    static const b2Vec2 points[] = {
        {0.007136f, 1.040076f}, {-0.119341f, 0.985289f},
        {-0.230611f, 0.667828f}, {-0.291870f, 0.082860f},
        {0.345968f, 0.082835f}, {0.289282f, 0.659785f},
        {0.154756f, 0.979975f},
    };
    b2Vec2 spawn = sim->level.count != 0 ? sim->level.spawn : b2Vec2_zero;
    sim->pot = create_dynamic_body(sim, spawn, 0.0f, 0.20f, 0.05f, false, true);
    b2ShapeDef shape = shape_definition(1.0f, 0.0f, 0xE6A95B);
    b2Hull hull = b2ComputeHull(points, 7);
    b2Polygon polygon = b2MakePolygon(&hull, 0.02f);
    b2CreatePolygonShape(sim->pot, &shape, &polygon);
    set_body_mass(sim->pot, 52.0f);
}

static void add_tip_shape(Sim *sim)
{
    /* Box2D's 5 mm linear tolerance rules out a lossless decomposition of
     * the source's 16-point concave outline. These are its stable extremes. */
    static const b2Vec2 extremes[] = {
        {-0.07085598f,  0.15148064f}, { 0.04644406f,  0.16873382f},
        { 0.07150281f,  0.15570971f}, { 0.07677472f, -0.09021313f},
        { 0.02481902f, -0.24082406f}, {-0.00929988f, -0.27907652f},
        {-0.02236808f, -0.23801441f}, {-0.06612671f, -0.08954985f},
    };
    b2ShapeDef shape = shape_definition(1.0f, 1.0f, 0xD7E5EA);
    b2Hull hull = b2ComputeHull(extremes, 8);
    b2Polygon polygon = b2MakePolygon(&hull, 0.0f);
    b2CreatePolygonShape(sim->tip, &shape, &polygon);
}

static void create_hammer(Sim *sim)
{
    b2Vec2 hub_position = b2Body_GetWorldPoint(sim->pot, player_hub_anchor);
    b2Rot slider_rotation = b2MakeRot(initial_slider_angle);
    b2Vec2 axis = b2RotateVector(slider_rotation, (b2Vec2){1.0f, 0.0f});
    b2Vec2 handle_position = b2MulAdd(
        hub_position, initial_slider_translation, axis);
    b2Vec2 pole_position = b2MulSub(handle_position, handle_to_pole, axis);
    b2Vec2 tip_position = b2MulSub(pole_position, pole_to_tip, axis);

    sim->hub = create_dynamic_body(sim, hub_position, 0.0f, 0.20f, 0.20f, true, false);
    sim->slider_body = create_dynamic_body(
        sim, hub_position, initial_slider_angle, 0.20f, 0.20f, false, false);
    sim->handle = create_dynamic_body(
        sim, handle_position, initial_slider_angle, 1.0f, 1.0f, false, false);
    sim->pole = create_dynamic_body(
        sim, pole_position, initial_slider_angle, 0.0f, 0.05f, false, false);
    sim->tip = create_dynamic_body(
        sim, tip_position, initial_slider_angle, 0.0f, 0.05f, false, true);

    set_body_mass(sim->hub, 10.0f);
    set_body_mass(sim->slider_body, 7.0f);
    set_body_mass(sim->handle, 3.0f);
    set_body_mass(sim->pole, 3.0f);

    /* The Unity hammer's Tip is the only hammer collision body. */
    add_tip_shape(sim);
    set_body_mass(sim->tip, 4.0f);
    sim->target = tip_position;
}

static void create_joints(Sim *sim)
{
    b2RevoluteJointDef pot_hub = b2DefaultRevoluteJointDef();
    pot_hub.bodyIdA = sim->pot;
    pot_hub.bodyIdB = sim->hub;
    pot_hub.localAnchorA = player_hub_anchor;
    pot_hub.localAnchorB = b2Vec2_zero;
    pot_hub.enableLimit = true;
    pot_hub.lowerAngle = -15.0f * B2_PI / 180.0f;
    pot_hub.upperAngle = 15.0f * B2_PI / 180.0f;
    sim->pot_hub = b2CreateRevoluteJoint(sim->world, &pot_hub);

    b2RevoluteJointDef hub_slider = b2DefaultRevoluteJointDef();
    hub_slider.bodyIdA = sim->hub;
    hub_slider.bodyIdB = sim->slider_body;
    hub_slider.localAnchorA = b2Vec2_zero;
    hub_slider.localAnchorB = b2Vec2_zero;
    hub_slider.referenceAngle = initial_slider_angle;
    hub_slider.enableMotor = true;
    hub_slider.maxMotorTorque = 10000.0f;
    sim->hub_slider = b2CreateRevoluteJoint(sim->world, &hub_slider);

    b2PrismaticJointDef slider_handle = b2DefaultPrismaticJointDef();
    slider_handle.bodyIdA = sim->slider_body;
    slider_handle.bodyIdB = sim->handle;
    slider_handle.localAnchorA = b2Vec2_zero;
    slider_handle.localAnchorB = b2Vec2_zero;
    slider_handle.localAxisA = (b2Vec2){1.0f, 0.0f};
    slider_handle.enableLimit = true;
    slider_handle.lowerTranslation = slider_lower_translation;
    slider_handle.upperTranslation = slider_upper_translation;
    slider_handle.enableMotor = true;
    slider_handle.maxMotorForce = 4300.0f;
    sim->slider_handle = b2CreatePrismaticJoint(sim->world, &slider_handle);

    b2WeldJointDef handle_pole = b2DefaultWeldJointDef();
    handle_pole.bodyIdA = sim->handle;
    handle_pole.bodyIdB = sim->pole;
    handle_pole.localAnchorA = b2Vec2_zero;
    handle_pole.localAnchorB = (b2Vec2){handle_to_pole, 0.0f};
    handle_pole.linearHertz = 70.0f;
    handle_pole.angularHertz = 70.0f;
    handle_pole.linearDampingRatio = 1.0f;
    handle_pole.angularDampingRatio = 1.0f;
    sim->handle_pole = b2CreateWeldJoint(sim->world, &handle_pole);

    b2WeldJointDef pole_tip = b2DefaultWeldJointDef();
    pole_tip.bodyIdA = sim->pole;
    pole_tip.bodyIdB = sim->tip;
    pole_tip.localAnchorA = (b2Vec2){-pole_to_tip, 0.0f};
    pole_tip.localAnchorB = b2Vec2_zero;
    sim->pole_tip = b2CreateWeldJoint(sim->world, &pole_tip);
}

static void create_world(Sim *sim)
{
    b2WorldDef world = b2DefaultWorldDef();
    world.gravity = (b2Vec2){0.0f, -9.81f};
    sim->world = b2CreateWorld(&world);
    create_level(sim);
    create_pot(sim);
    create_hammer(sim);
    create_joints(sim);
    sim->max_height = b2Body_GetPosition(sim->pot).y;
    sim->tick = 0;
}

static void free_level(Level *level)
{
    for (size_t i = 0; i < level->count; ++i) {
        free(level->shapes[i].points);
    }
    free(level->shapes);
    *level = (Level){0};
}

Sim *sim_create(void)
{
    Sim *sim = calloc(1, sizeof(*sim));
    if (sim != NULL) {
        create_world(sim);
    }
    return sim;
}

void sim_destroy(Sim *sim)
{
    if (sim != NULL) {
        b2DestroyWorld(sim->world);
        free_level(&sim->level);
        free(sim);
    }
}

static bool valid_point(b2Vec2 point)
{
    return isfinite(point.x) && isfinite(point.y)
        && fabsf(point.x) < 1000000.0f && fabsf(point.y) < 1000000.0f;
}

static void extend_bounds(Level *level, b2Vec2 point)
{
    level->min.x = fminf(level->min.x, point.x);
    level->min.y = fminf(level->min.y, point.y);
    level->max.x = fmaxf(level->max.x, point.x);
    level->max.y = fmaxf(level->max.y, point.y);
}

static bool read_polygon(FILE *file, LevelShape *shape)
{
    if (fscanf(file, "%d", &shape->count) != 1 || shape->count < 3 || shape->count > 100000) {
        return false;
    }
    shape->points = malloc((size_t)shape->count * sizeof(*shape->points));
    if (shape->points == NULL) {
        return false;
    }
    for (int i = 0; i < shape->count; ++i) {
        b2Vec2 *point = &shape->points[i];
        if (fscanf(file, "%f %f", &point->x, &point->y) != 2 || !valid_point(*point)) {
            return false;
        }
    }
    if (b2DistanceSquared(shape->points[0], shape->points[shape->count - 1]) == 0.0f) {
        --shape->count; /* Only the exactly duplicated closing vertex. */
    }
    if (shape->count < 3) {
        return false;
    }
    double area = 0.0;
    for (int i = 0; i < shape->count; ++i) {
        b2Vec2 a = shape->points[i];
        b2Vec2 b = shape->points[(i + 1) % shape->count];
        if (b2DistanceSquared(a, b) == 0.0f) {
            return false;
        }
        area += (double)a.x * b.y - (double)b.x * a.y;
    }
    if (area == 0.0) {
        return false;
    }
    if (area < 0.0) {
        for (int i = 0; i < shape->count / 2; ++i) {
            b2Vec2 point = shape->points[i];
            shape->points[i] = shape->points[shape->count - 1 - i];
            shape->points[shape->count - 1 - i] = point;
        }
    }
    if (shape->count == 3 && b2ComputeHull(shape->points, 3).count != 3) {
        return false;
    }
    return true;
}

static bool read_shape(FILE *file, const char *kind, Level *level)
{
    int64_t id;
    LevelShape shape = {0};
    if (fscanf(file, "%" SCNd64 " %f %f", &id, &shape.friction, &shape.restitution) != 3
        || !isfinite(shape.friction) || shape.friction < 0.0f
        || !isfinite(shape.restitution) || shape.restitution < 0.0f) {
        return false;
    }
    bool ok = false;
    if (strcmp(kind, "polygon") == 0) {
        ok = read_polygon(file, &shape);
    } else if (strcmp(kind, "circle") == 0) {
        ok = fscanf(file, "%f %f %f", &shape.center.x, &shape.center.y, &shape.radius) == 3
            && valid_point(shape.center) && isfinite(shape.radius) && shape.radius > 0.0f;
    }
    if (!ok || level->count >= 100000) {
        free(shape.points);
        return false;
    }
    LevelShape *shapes = realloc(level->shapes, (level->count + 1) * sizeof(*shapes));
    if (shapes == NULL) {
        free(shape.points);
        return false;
    }
    level->shapes = shapes;
    level->shapes[level->count++] = shape;
    if (shape.radius > 0.0f) {
        extend_bounds(level, b2Sub(shape.center, (b2Vec2){shape.radius, shape.radius}));
        extend_bounds(level, b2Add(shape.center, (b2Vec2){shape.radius, shape.radius}));
    } else {
        for (int i = 0; i < shape.count; ++i) {
            extend_bounds(level, shape.points[i]);
        }
    }
    return true;
}

static bool read_level(FILE *file, Level *level)
{
    char word[32];
    int version;
    if (fscanf(file, "%31s %d", word, &version) != 2
        || strcmp(word, "AIGET_LEVEL") != 0 || version != 1) {
        return false;
    }
    bool has_spawn = false;
    level->min = (b2Vec2){INFINITY, INFINITY};
    level->max = (b2Vec2){-INFINITY, -INFINITY};
    while (fscanf(file, "%31s", word) == 1) {
        if (strcmp(word, "spawn") == 0) {
            if (has_spawn || fscanf(file, "%f %f", &level->spawn.x, &level->spawn.y) != 2
                || !valid_point(level->spawn)) {
                return false;
            }
            has_spawn = true;
        } else if (!read_shape(file, word, level)) {
            return false;
        }
    }
    return has_spawn && level->count > 0 && !ferror(file);
}

bool sim_load_level(Sim *sim, const char *path, char *error, size_t error_size)
{
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "Cannot open collision map: %s", path);
        }
        return false;
    }
    Level level = {0};
    bool ok = read_level(file, &level);
    fclose(file);
    if (!ok) {
        free_level(&level);
        if (error != NULL && error_size > 0) {
            snprintf(error, error_size, "Malformed/degenerate AIGET_LEVEL 1 map: %s", path);
        }
        return false;
    }
    b2DestroyWorld(sim->world);
    free_level(&sim->level);
    sim->level = level;
    create_world(sim);
    if (error != NULL && error_size > 0) {
        error[0] = '\0';
    }
    return true;
}

const char *sim_level_name(const Sim *sim)
{
    return sim->level.count != 0 ? "Imported collision map" : "Practice course";
}

void sim_reset(Sim *sim)
{
    /* Rebuild contacts and solver caches too, not just visible transforms. */
    b2DestroyWorld(sim->world);
    create_world(sim);
}

static void update_target(Sim *sim, float dx, float dy)
{
    if (!isfinite(dx) || !isfinite(dy)) {
        return;
    }
    sim->target.x += clamp(dx, -10000.0f, 10000.0f) * mouse_scale;
    sim->target.y -= clamp(dy, -10000.0f, 10000.0f) * mouse_scale;
}

static b2Vec2 reachable_target(Sim *sim)
{
    b2Vec2 pivot = b2Body_GetPosition(sim->hub);
    b2Vec2 delta = b2Sub(sim->target, pivot);
    float length = b2Length(delta);
    float min_length = tip_reach_offset - slider_upper_translation;
    float max_length = tip_reach_offset - slider_lower_translation;
    if (length < 0.0001f) {
        delta = b2RotateVector(b2Body_GetRotation(sim->slider_body), (b2Vec2){-1.0f, 0.0f});
        length = 1.0f;
    }
    float clamped_length = clamp(length, min_length, max_length);
    sim->target = b2Add(pivot, b2MulSV(clamped_length / length, delta));
    return b2Sub(sim->target, pivot);
}

static void update_motors(Sim *sim)
{
    b2Vec2 target = reachable_target(sim);
    float target_length = b2Length(target);
    float hub_angle = b2Rot_GetAngle(b2Body_GetRotation(sim->hub));
    float desired_slider_angle = atan2f(target.y, target.x) - B2_PI;
    float desired_joint_angle = desired_slider_angle - hub_angle - initial_slider_angle;
    float current_joint_angle = b2RevoluteJoint_GetAngle(sim->hub_slider);
    float turn_speed = 18.0f * angle_error(desired_joint_angle, current_joint_angle);
    b2RevoluteJoint_SetMotorSpeed(sim->hub_slider, clamp(turn_speed, -10.0f, 10.0f));

    float desired_translation = tip_reach_offset - target_length;
    float extension = b2PrismaticJoint_GetTranslation(sim->slider_handle);
    float slide_speed = 16.0f * (desired_translation - extension);
    b2PrismaticJoint_SetMotorSpeed(sim->slider_handle, clamp(slide_speed, -6.0f, 6.0f));
}

void sim_step(Sim *sim, float mouse_dx, float mouse_dy)
{
    update_target(sim, mouse_dx, mouse_dy);
    update_motors(sim);
    b2World_Step(sim->world, SIM_TIMESTEP, 4);
    sim->max_height = fmaxf(sim->max_height, b2Body_GetPosition(sim->pot).y);
    ++sim->tick;
}

static int contact_count(b2BodyId body)
{
    b2ContactData contacts[32];
    int count = b2Body_GetContactData(body, contacts, 32);
    int touching = 0;
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < contacts[i].manifold.pointCount; ++j) {
            if (contacts[i].manifold.points[j].separation <= 0.005f) {
                ++touching;
                break;
            }
        }
    }
    return touching;
}

SimSnapshot sim_snapshot(const Sim *sim)
{
    b2Vec2 pivot = b2Body_GetPosition(sim->hub);
    float finish = sim->level.count != 0 ? sim->level.max.y : SIM_FINISH_HEIGHT;
    return (SimSnapshot){
        .tick = sim->tick,
        .pot_position = b2Body_GetPosition(sim->pot),
        .pot_velocity = b2Body_GetLinearVelocity(sim->pot),
        .hammer_position = b2Body_GetPosition(sim->tip),
        .hammer_velocity = b2Body_GetLinearVelocity(sim->tip),
        .pivot_position = pivot,
        .target_position = sim->target,
        .level_min = sim->level.count != 0 ? sim->level.min : (b2Vec2){-12.0f, -1.0f},
        .level_max = sim->level.count != 0 ? sim->level.max : (b2Vec2){8.0f, 12.0f},
        .pot_angle = b2Rot_GetAngle(b2Body_GetRotation(sim->pot)),
        .hammer_angle = b2Rot_GetAngle(b2Body_GetRotation(sim->tip)),
        .extension = b2PrismaticJoint_GetTranslation(sim->slider_handle),
        .max_height = sim->max_height,
        .finish_height = finish,
        .hammer_contacts = contact_count(sim->tip),
        .pot_contacts = contact_count(sim->pot),
        .finished = sim->max_height >= finish,
        .has_external_level = sim->level.count != 0,
    };
}

b2WorldId sim_world(const Sim *sim)
{
    return sim->world;
}
