/* Read-only importer for this owned game's Unity 2020.3.6f1 serialized scene.
 * It intentionally rejects other layouts instead of guessing their offsets. */
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint8_t *bytes; size_t size, at; } Reader;
typedef struct { int64_t id; int type; size_t offset, size; } Object;
typedef struct { Reader data; Object *objects; int count; } Scene;
typedef struct { double x, y, z; } Point;
typedef struct { int64_t transform, body; unsigned layer; bool active; char name[128]; } GameObject;
typedef struct { int64_t game_object, parent; double q[4]; Point position, scale; } Transform;
typedef struct { unsigned paths, circles, vertices, skipped, unsupported; double min_edge; } Stats;

static void fail(const char *message) { fprintf(stderr, "export: %s\n", message); exit(1); }
static void require(bool condition, const char *message) { if (!condition) fail(message); }
static void skip(Reader *r, size_t n) { require(r->at <= r->size && n <= r->size-r->at, "truncated asset"); r->at += n; }
static uint64_t integer(Reader *r, unsigned n, bool big) {
    uint64_t v = 0; size_t at = r->at; skip(r, n);
    for (unsigned i = 0; i < n; ++i) v |= (uint64_t)r->bytes[at+i] << (8*(big ? n-i-1 : i));
    return v;
}
static uint32_t u32(Reader *r) { return (uint32_t)integer(r, 4, false); }
static int64_t i64(Reader *r) { return (int64_t)integer(r, 8, false); }
static float f32(Reader *r) { uint32_t bits=u32(r); float v; memcpy(&v,&bits,4); require(isfinite(v),"non-finite geometry"); return v; }
static void align4(Reader *r) { skip(r, (4-r->at%4)%4); }
static int64_t pointer(Reader *r) { require(u32(r)==0, "external scene-object pointer"); return i64(r); }
static void string(Reader *r, char *out, size_t capacity) {
    unsigned n=u32(r); size_t at=r->at; skip(r,n);
    if (capacity) { size_t take=n<capacity-1 ? n : capacity-1; memcpy(out,r->bytes+at,take); out[take]=0; }
    align4(r);
}
static Reader load(const char *path) {
    FILE *f=fopen(path,"rb"); require(f!=NULL,"cannot open input asset");
    require(fseek(f,0,SEEK_END)==0,"cannot seek asset"); long length=ftell(f);
    require(length>48,"asset too small"); rewind(f);
    Reader r={.size=(size_t)length}; r.bytes=malloc(r.size); require(r.bytes!=NULL,"out of memory");
    require(fread(r.bytes,1,r.size,f)==r.size,"cannot read asset"); fclose(f); return r;
}
static int object_order(const void *a, const void *b) {
    int64_t x=((const Object *)a)->id,y=((const Object *)b)->id; return (x>y)-(x<y);
}
static Scene scene_load(const char *path) {
    Scene s={.data=load(path)}; Reader *r=&s.data;
    skip(r,8); require(integer(r,4,true)==22,"expected serialized version 22"); skip(r,4);
    require(integer(r,1,false)==0,"expected little-endian asset"); skip(r,7);
    require(integer(r,8,true)==r->size,"asset length mismatch"); size_t data_offset=(size_t)integer(r,8,true); skip(r,8);
    const char version[]="2020.3.6f1"; require(r->size-r->at>sizeof(version),"missing Unity version");
    require(memcmp(r->bytes+r->at,version,sizeof(version))==0,"unsupported Unity version"); skip(r,sizeof(version)+4);
    require(integer(r,1,false)==0,"expected stripped type trees"); unsigned count=u32(r);
    require(count>0 && count<10000,"invalid type count"); int *types=calloc(count,sizeof(*types)); require(types!=NULL,"out of memory");
    for (unsigned i=0;i<count;++i) { types[i]=(int)u32(r); skip(r,3+(types[i]==114 ? 32:16)); }
    s.count=(int)u32(r); require(s.count>0 && s.count<1000000,"invalid object count");
    s.objects=calloc((size_t)s.count,sizeof(*s.objects)); require(s.objects!=NULL,"out of memory");
    for (int i=0;i<s.count;++i) {
        align4(r); Object *o=&s.objects[i]; o->id=i64(r); o->offset=data_offset+(size_t)i64(r); o->size=u32(r);
        unsigned type=u32(r); require(type<count,"invalid type index"); o->type=types[type];
        require(o->offset<=r->size && o->size<=r->size-o->offset,"object outside asset");
    }
    free(types); qsort(s.objects,(size_t)s.count,sizeof(*s.objects),object_order); return s;
}
static Object *find(Scene *s, int64_t id) { Object key={.id=id}; return bsearch(&key,s->objects,(size_t)s->count,sizeof(key),object_order); }
static Reader object_reader(Scene *s, Object *o) { require(o!=NULL,"missing scene object"); return (Reader){s->data.bytes+o->offset,o->size,0}; }
static GameObject game_object(Scene *s, int64_t id) {
    Object *o=find(s,id); require(o && o->type==1,"missing GameObject"); Reader r=object_reader(s,o); GameObject g={0};
    unsigned count=u32(&r); require(count<=1000,"invalid component count");
    for (unsigned i=0;i<count;++i) { Object *c=find(s,pointer(&r)); if (!c) continue; if(c->type==4)g.transform=c->id; if(c->type==50)g.body=c->id; }
    g.layer=u32(&r); string(&r,g.name,sizeof(g.name)); skip(&r,2); g.active=integer(&r,1,false)!=0; return g;
}
static Point point(Reader *r) { Point p; p.x=f32(r);p.y=f32(r);p.z=f32(r);return p; }
static Transform transform(Scene *s, int64_t id) {
    Object *o=find(s,id); require(o && o->type==4,"missing Transform"); Reader r=object_reader(s,o); Transform t={0};
    t.game_object=pointer(&r); for(int i=0;i<4;++i)t.q[i]=f32(&r); t.position=point(&r);t.scale=point(&r);
    unsigned children=u32(&r); require(children<100000,"invalid child count"); skip(&r,(size_t)children*12); t.parent=pointer(&r); return t;
}
static Point transform_point(Transform t, Point p) {
    double x=t.q[0],y=t.q[1],z=t.q[2],w=t.q[3]; p.x*=t.scale.x;p.y*=t.scale.y;p.z*=t.scale.z;
    return (Point){(1-2*(y*y+z*z))*p.x+2*(x*y-z*w)*p.y+2*(x*z+y*w)*p.z+t.position.x,
        2*(x*y+z*w)*p.x+(1-2*(x*x+z*z))*p.y+2*(y*z-x*w)*p.z+t.position.y,
        2*(x*z-y*w)*p.x+2*(y*z+x*w)*p.y+(1-2*(x*x+y*y))*p.z+t.position.z};
}
static Point world_point(Scene *s,int64_t id,Point p) {
    unsigned depth=0; while(id) { require(++depth<256,"transform cycle"); Transform t=transform(s,id);p=transform_point(t,p);id=t.parent; } return p;
}
static bool is_static_active(Scene *s,GameObject g) {
    unsigned depth=0;
    while(g.transform) {
        require(++depth<256,"transform cycle"); if(!g.active)return false;
        if(g.body) { Reader r=object_reader(s,find(s,g.body));skip(&r,12); if(u32(&r)!=2)return false; }
        Transform t=transform(s,g.transform); if(!t.parent)break; g=game_object(s,transform(s,t.parent).game_object);
    }
    return true;
}
static void material(Scene *assets,int file,int64_t id,float *friction,float *bounce) {
    if(id==0) { *friction=.4f;*bounce=0;return; }
    require(file==2,"unsupported material asset reference"); Object *o=find(assets,id);require(o && o->type==62,"missing PhysicsMaterial2D");
    Reader r=object_reader(assets,o);string(&r,NULL,0);*friction=f32(&r);*bounce=f32(&r);
}
static void polygon(FILE *out,int64_t id,float friction,float bounce,Point *points,unsigned n,Stats *stats) {
    require(n>=3,"polygon has fewer than three vertices"); fprintf(out,"polygon %" PRId64 " %.9g %.9g %u\n",id,friction,bounce,n);
    for(unsigned i=0;i<n;++i) {
        fprintf(out,"%.9g %.9g\n",points[i].x,points[i].y); Point next=points[(i+1)%n];
        double length=hypot(points[i].x-next.x,points[i].y-next.y); if(length<stats->min_edge)stats->min_edge=length;
    }
    ++stats->paths;stats->vertices+=n;
}
static void collider(Scene *s,Scene *assets,Object *o,FILE *out,Stats *stats) {
    Reader r=object_reader(s,o); GameObject g=game_object(s,pointer(&r)); bool enabled=integer(&r,1,false)!=0;
    align4(&r);skip(&r,4);int mat_file=(int)u32(&r);int64_t mat_id=i64(&r);
    bool trigger=integer(&r,1,false)!=0,effector=integer(&r,1,false)!=0,composite=integer(&r,1,false)!=0;align4(&r);
    Point offset={0};offset.x=f32(&r);offset.y=f32(&r);
    if(!enabled || trigger || !is_static_active(s,g)) { ++stats->skipped;return; }
    if(effector || composite || (g.layer!=0 && g.layer!=10)) {
        fprintf(stderr,"unsupported collider %" PRId64 " (%s): layer=%u effector=%d composite=%d\n",o->id,g.name,g.layer,effector,composite);++stats->unsupported;return;
    }
    float friction,bounce;material(assets,mat_file,mat_id,&friction,&bounce);
    if(o->type==58) {
        double radius=f32(&r);Point center=world_point(s,g.transform,offset);
        Point x=world_point(s,g.transform,(Point){offset.x+radius,offset.y,0});
        Point y=world_point(s,g.transform,(Point){offset.x,offset.y+radius,0});
        double rx=hypot(x.x-center.x,x.y-center.y),ry=hypot(y.x-center.x,y.y-center.y);
        if(fabs(rx-ry)>1e-5*fmax(1,rx))fprintf(stderr,"circle %" PRId64 " (%s): nonuniform scale; Unity maximum-axis radius used\n",o->id,g.name);
        fprintf(out,"circle %" PRId64 " %.9g %.9g %.9g %.9g %.9g\n",o->id,friction,bounce,center.x,center.y,fmax(rx,ry));++stats->circles;return;
    }
    skip(&r,49);align4(&r);bool tiling=integer(&r,1,false)!=0;align4(&r);
    if(tiling) { fprintf(stderr,"unsupported auto-tiling collider %" PRId64 "\n",o->id);++stats->unsupported;return; }
    if(o->type==61) {
        double hx=f32(&r)/2,hy=f32(&r)/2,radius=f32(&r);
        if(radius!=0) { fprintf(stderr,"unsupported rounded box %" PRId64 "\n",o->id);++stats->unsupported;return; }
        Point p[4]={{offset.x-hx,offset.y-hy,0},{offset.x+hx,offset.y-hy,0},{offset.x+hx,offset.y+hy,0},{offset.x-hx,offset.y+hy,0}};
        for (int i = 0; i < 4; ++i) {
            p[i] = world_point(s, g.transform, p[i]);
        }
        polygon(out, o->id, friction, bounce, p, 4, stats);
        return;
    }
    unsigned paths=u32(&r);require(paths>0 && paths<10000,"invalid polygon path count");
    if(paths>1)fprintf(stderr,"multi-path collider %" PRId64 " (%s): %u ordered paths retained under same ID\n",o->id,g.name,paths);
    for(unsigned j=0;j<paths;++j) {
        unsigned n=u32(&r);require(n>=3 && n<100000,"invalid polygon vertex count");Point *p=calloc(n,sizeof(*p));require(p!=NULL,"out of memory");
        for(unsigned i=0;i<n;++i){p[i].x=f32(&r)+offset.x;p[i].y=f32(&r)+offset.y;p[i]=world_point(s,g.transform,p[i]);}
        polygon(out,o->id,friction,bounce,p,n,stats);free(p);align4(&r);
    }
    require(r.at==r.size,"unexpected collider layout");
}
static Point spawn(Scene *s) {
    for(int i=0;i<s->count;++i)if(s->objects[i].type==50) {
        Reader r=object_reader(s,&s->objects[i]);GameObject g=game_object(s,pointer(&r));
        if(strcmp(g.name,"Player")==0)return world_point(s,g.transform,(Point){0});
    }
    fail("Player rigid body not found");return (Point){0};
}
int main(int argc,char **argv) {
    if(argc!=4) { fprintf(stderr,"usage: %s LEVEL1 SHAREDASSETS1 OUTPUT\n",argv[0]);return 2; }
    Scene scene=scene_load(argv[1]),assets=scene_load(argv[2]);Point start=spawn(&scene);
    FILE *out=fopen(argv[3],"wx");require(out!=NULL,"cannot create output (existing files are never overwritten)");
    fprintf(out,"AIGET_LEVEL 1\nspawn %.9g %.9g\n",start.x,start.y);Stats stats={.min_edge=INFINITY};
    for(int i=0;i<scene.count;++i){Object *o=&scene.objects[i];if(o->type==58 || o->type==60 || o->type==61)collider(&scene,&assets,o,out,&stats);}
    require(fclose(out)==0,"cannot finish output");
    fprintf(stderr,"exported %u paths, %u circles, %u vertices; skipped %u inactive/trigger/moving colliders; unsupported %u; minimum edge %.9g; spawn %.9g %.9g\n",stats.paths,stats.circles,stats.vertices,stats.skipped,stats.unsupported,stats.min_edge,start.x,start.y);
    free(scene.objects);free(scene.data.bytes);free(assets.objects);free(assets.data.bytes);
    return stats.unsupported ? 3:0;
}
