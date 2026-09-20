#include "view.h"

#include <raylib.h>

/* Xlib's legacy Font typedef collides with raylib's public Font struct. */
#define Font X11Font
#include <X11/Xlib.h>
#undef Font

#include <math.h>
#include <stdio.h>

enum {
    screen_width = 1280,
    screen_height = 720,
    max_draw_vertices = 32,
};

typedef struct {
    Camera2D camera;
} DrawContext;

static Vector2 screen_point(b2Vec2 point)
{
    return (Vector2){point.x, -point.y};
}

static Color draw_color(b2HexColor color)
{
    return (Color){
        .r = (unsigned char)((color >> 16) & 0xFF),
        .g = (unsigned char)((color >> 8) & 0xFF),
        .b = (unsigned char)(color & 0xFF),
        .a = 255,
    };
}

static void draw_polygon(const b2Vec2 *vertices, int count, b2HexColor color, void *context)
{
    (void)context;
    if (count < 2 || count > max_draw_vertices) {
        return;
    }
    Vector2 points[max_draw_vertices];
    for (int i = 0; i < count; ++i) {
        points[i] = screen_point(vertices[i]);
    }
    for (int i = 1; i + 1 < count; ++i) {
        DrawTriangle(points[0], points[i], points[i + 1], Fade(draw_color(color), 0.28f));
    }
    for (int i = 0; i < count; ++i) {
        DrawLineEx(points[i], points[(i + 1) % count], 0.025f, draw_color(color));
    }
}

static void draw_solid_polygon(
    b2Transform transform,
    const b2Vec2 *vertices,
    int count,
    float radius,
    b2HexColor color,
    void *context)
{
    (void)radius;
    if (count < 2 || count > max_draw_vertices) {
        return;
    }
    b2Vec2 world_vertices[max_draw_vertices];
    for (int i = 0; i < count; ++i) {
        world_vertices[i] = b2TransformPoint(transform, vertices[i]);
    }
    draw_polygon(world_vertices, count, color, context);
}

static void draw_circle(b2Vec2 center, float radius, b2HexColor color, void *context)
{
    (void)context;
    DrawCircleV(screen_point(center), radius, Fade(draw_color(color), 0.28f));
    DrawCircleLinesV(screen_point(center), radius, draw_color(color));
}

static void draw_solid_circle(b2Transform transform, float radius, b2HexColor color, void *context)
{
    (void)context;
    draw_circle(transform.p, radius, color, NULL);
}

static void draw_capsule(b2Vec2 first, b2Vec2 second, float radius, b2HexColor color, void *context)
{
    (void)context;
    Color fill = Fade(draw_color(color), 0.28f);
    DrawLineEx(screen_point(first), screen_point(second), radius * 2.0f, fill);
    DrawCircleV(screen_point(first), radius, fill);
    DrawCircleV(screen_point(second), radius, fill);
    DrawLineEx(screen_point(first), screen_point(second), 0.025f, draw_color(color));
}

static void draw_segment(b2Vec2 first, b2Vec2 second, b2HexColor color, void *context)
{
    (void)context;
    DrawLineEx(screen_point(first), screen_point(second), 0.025f, draw_color(color));
}

static b2DebugDraw debug_draw(void)
{
    b2DebugDraw draw = b2DefaultDebugDraw();
    draw.DrawPolygonFcn = draw_polygon;
    draw.DrawSolidPolygonFcn = draw_solid_polygon;
    draw.DrawCircleFcn = draw_circle;
    draw.DrawSolidCircleFcn = draw_solid_circle;
    draw.DrawSolidCapsuleFcn = draw_capsule;
    draw.DrawSegmentFcn = draw_segment;
    draw.drawShapes = true;
    draw.drawJoints = false;
    return draw;
}

static Camera2D game_camera(const GameView *view, SimSnapshot state)
{
    return (Camera2D){
        .offset = {(float)screen_width * 0.42f, (float)screen_height * 0.67f},
        .target = {state.pot_position.x, -state.pot_position.y},
        .rotation = 0.0f,
        .zoom = view->zoom,
    };
}

static void draw_target(SimSnapshot state)
{
    Vector2 pivot = screen_point(state.pivot_position);
    Vector2 target = screen_point(state.target_position);
    DrawLineEx(pivot, target, 0.018f, Fade(RED, 0.7f));
    DrawCircleLinesV(target, 0.10f, RED);
    DrawLineEx((Vector2){target.x - 0.12f, target.y}, (Vector2){target.x + 0.12f, target.y}, 0.018f, RED);
    DrawLineEx((Vector2){target.x, target.y - 0.12f}, (Vector2){target.x, target.y + 0.12f}, 0.018f, RED);
}

static void draw_hud(const GameView *view, SimSnapshot state)
{
    char state_line[160];
    snprintf(
        state_line,
        sizeof(state_line),
        "tick %llu   height %.2f / %.2f   hammer contacts %d   pot contacts %d",
        (unsigned long long)state.tick,
        state.max_height,
        state.finish_height,
        state.hammer_contacts,
        state.pot_contacts);
    DrawRectangle(12, 12, 590, view->show_help ? 105 : 32, Fade(BLACK, 0.55f));
    DrawText(state_line, 20, 20, 18, RAYWHITE);
    if (view->show_help) {
        DrawText("Move mouse: move the hammer target", 20, 48, 18, RAYWHITE);
        DrawText("R: reset   H: hide help   wheel: zoom   Esc: quit", 20, 74, 18, RAYWHITE);
    }
    if (state.finished) {
        DrawText("COURSE HEIGHT REACHED", screen_width / 2 - 150, 42, 28, GOLD);
    }
}

bool view_open(GameView *view)
{
    Display *display = XOpenDisplay(NULL);
    if (display == NULL) {
        fputs("Could not open DISPLAY. Start the game from a graphical desktop session.\n", stderr);
        return false;
    }
    XCloseDisplay(display);

    InitWindow(screen_width, screen_height, "AIget - C physics engine");
    if (!IsWindowReady()) {
        fputs("Could not open a raylib window. Set a working DISPLAY or start a desktop session.\n", stderr);
        return false;
    }
    SetTargetFPS(120);
    DisableCursor();
    if (view->zoom <= 0.0f) {
        view->zoom = 78.0f;
    }
    return true;
}

void view_close(void)
{
    EnableCursor();
    CloseWindow();
}

bool view_should_close(void)
{
    return WindowShouldClose();
}

float view_frame_seconds(void)
{
    return fminf(GetFrameTime(), 0.10f);
}

GameInput view_read_input(GameView *view)
{
    if (IsKeyPressed(KEY_H)) {
        view->show_help = !view->show_help;
    }
    float wheel = GetMouseWheelMove();
    view->zoom = fmaxf(24.0f, fminf(160.0f, view->zoom * (1.0f + wheel * 0.12f)));
    Vector2 mouse = GetMouseDelta();
    return (GameInput){
        .mouse_dx = mouse.x,
        .mouse_dy = mouse.y,
        .reset = IsKeyPressed(KEY_R),
    };
}

void view_draw(GameView *view, const Sim *sim)
{
    SimSnapshot state = sim_snapshot(sim);
    DrawContext context = {.camera = game_camera(view, state)};
    b2DebugDraw draw = debug_draw();
    draw.context = &context;

    BeginDrawing();
    ClearBackground((Color){20, 25, 32, 255});
    BeginMode2D(context.camera);
    b2World_Draw(sim_world(sim), &draw);
    draw_target(state);
    EndMode2D();
    draw_hud(view, state);
    EndDrawing();
}
