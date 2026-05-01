#pragma once
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <string>
#include <vector>
#include <functional>
#include <cstdio>
#include <algorithm>
#include <cmath>

struct UIDragState { int x, y; bool held; bool just_pressed; bool prev_mouse_down; };

struct UIContext {
  TTF_Font *font;
  SDL_Renderer *renderer;
  int mouse_x, mouse_y;
  bool mouse_down;
  bool mouse_clicked;
  SDL_Keymod mods;
  UIDragState *drag;
};

static void ui_render_text(SDL_Renderer *r, TTF_Font *f, const char *text,
                           SDL_Color c, int x, int y) {
  if (!f || !text) return;
  SDL_Surface *surf = TTF_RenderText_Solid(f, text, c);
  if (!surf) return;
  SDL_Texture *tex = SDL_CreateTextureFromSurface(r, surf);
  if (tex) {
    SDL_Rect d{x, y, surf->w, surf->h};
    SDL_RenderCopy(r, tex, nullptr, &d);
    SDL_DestroyTexture(tex);
  }
  SDL_FreeSurface(surf);
}

struct UIWidget {
  SDL_Rect rect{};
  bool visible = true;
  virtual ~UIWidget() = default;
  virtual void handle(UIContext &ctx) {}
  virtual void draw(UIContext &ctx) = 0;
};

struct UISlider : UIWidget {
  std::string label;
  float *value = nullptr;
  float min_val = 0, max_val = 100;
  float step = 1;
  SDL_Color track{60,70,90,255};
  SDL_Color fill{100,160,220,255};
  SDL_Color thumb{180,210,240,255};
  SDL_Color label_color{200,210,220,255};

  void handle(UIContext &ctx) override {
    if (!visible || !value) return;
    bool hover = ctx.mouse_y >= rect.y && ctx.mouse_y < rect.y + rect.h &&
                 ctx.mouse_x >= rect.x - 4 && ctx.mouse_x < rect.x + rect.w + 4;
    if (!hover) return;
    if (ctx.mouse_down) {
      ctx.drag->held = true;
      ctx.drag->x = ctx.mouse_x;
    }
    if (ctx.drag->held) {
      float t = (float)(ctx.mouse_x - rect.x) / rect.w;
      t = std::clamp(t, 0.0f, 1.0f);
      float v = min_val + t * (max_val - min_val);
      if (step > 0) v = std::round(v / step) * step;
      *value = std::clamp(v, min_val, max_val);
    }
  }

  void draw(UIContext &ctx) override {
    if (!visible || !value) return;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%s: %.0f", label.c_str(), *value);
    ui_render_text(ctx.renderer, ctx.font, buf, label_color, rect.x,
                   rect.y - 14);

    int sy = rect.y;
    int sh = rect.h;
    SDL_SetRenderDrawColor(ctx.renderer, track.r, track.g, track.b, 255);
    SDL_Rect tr{rect.x, sy + sh / 2 - 3, rect.w, 6};
    SDL_RenderFillRect(ctx.renderer, &tr);

    float t = (*value - min_val) / (max_val - min_val);
    t = std::clamp(t, 0.0f, 1.0f);
    int fx = rect.x + (int)(t * rect.w);
    SDL_SetRenderDrawColor(ctx.renderer, fill.r, fill.g, fill.b, 255);
    SDL_Rect fr{rect.x, sy + sh / 2 - 3, fx - rect.x, 6};
    if (fr.w > 0) SDL_RenderFillRect(ctx.renderer, &fr);

    SDL_SetRenderDrawColor(ctx.renderer, thumb.r, thumb.g, thumb.b, 255);
    SDL_Rect thr{fx - 5, sy, 10, sh};
    SDL_RenderFillRect(ctx.renderer, &thr);
  }
};

struct UIToggle : UIWidget {
  std::string label;
  bool *value = nullptr;
  SDL_Color off_bg{60,60,80,255};
  SDL_Color on_bg{80,160,100,255};
  SDL_Color label_color{200,210,220,255};
  std::function<void(bool)> on_change;

  void handle(UIContext &ctx) override {
    if (!visible || !value) return;
    bool hover = ctx.mouse_x >= rect.x && ctx.mouse_x < rect.x + rect.w &&
                 ctx.mouse_y >= rect.y && ctx.mouse_y < rect.y + rect.h;
    if (hover && ctx.mouse_clicked) {
      ctx.mouse_clicked = false;
      *value = !*value;
      if (on_change) on_change(*value);
    }
  }

  void draw(UIContext &ctx) override {
    if (!visible || !value) return;
    auto &bg = *value ? on_bg : off_bg;
    int tw = rect.w / 2;

    SDL_SetRenderDrawColor(ctx.renderer, bg.r, bg.g, bg.b, 255);
    SDL_RenderFillRect(ctx.renderer, &rect);
    SDL_SetRenderDrawColor(ctx.renderer, 120,140,170,255);
    SDL_RenderDrawRect(ctx.renderer, &rect);

    int knob_x = *value ? rect.x + tw : rect.x;
    SDL_SetRenderDrawColor(ctx.renderer, 200,210,220,255);
    SDL_Rect kn{knob_x + 2, rect.y + 2, tw - 4, rect.h - 4};
    SDL_RenderFillRect(ctx.renderer, &kn);

    ui_render_text(ctx.renderer, ctx.font, label.c_str(), label_color,
                   rect.x + rect.w + 8, rect.y + (rect.h - 14) / 2);
  }
};

struct UISelect : UIWidget {
  std::string label;
  int *value = nullptr;
  std::vector<std::string> options;
  SDL_Color bg{50,60,80,255};
  SDL_Color fg{220,230,240,255};
  SDL_Color label_color{200,210,220,255};

  void handle(UIContext &ctx) override {
    if (!visible || !value || options.empty()) return;
    bool hover = ctx.mouse_x >= rect.x && ctx.mouse_x < rect.x + rect.w &&
                 ctx.mouse_y >= rect.y && ctx.mouse_y < rect.y + rect.h;
    if (hover && ctx.mouse_clicked) {
      ctx.mouse_clicked = false;
      *value = (*value + 1) % (int)options.size();
    }
  }

  void draw(UIContext &ctx) override {
    if (!visible || !value || options.empty()) return;
    ui_render_text(ctx.renderer, ctx.font, label.c_str(), label_color,
                   rect.x, rect.y - 14);

    SDL_SetRenderDrawColor(ctx.renderer, bg.r, bg.g, bg.b, 255);
    SDL_RenderFillRect(ctx.renderer, &rect);
    SDL_SetRenderDrawColor(ctx.renderer, 120,140,170,255);
    SDL_RenderDrawRect(ctx.renderer, &rect);

    if (*value >= 0 && *value < (int)options.size()) {
      ui_render_text(ctx.renderer, ctx.font, options[*value].c_str(), fg,
                     rect.x + (rect.w - (int)options[*value].size() * 7) / 2,
                     rect.y + (rect.h - 14) / 2);
    }

    SDL_SetRenderDrawColor(ctx.renderer, 100,120,150,255);
    SDL_RenderDrawLine(ctx.renderer, rect.x + 4, rect.y + rect.h / 2 - 3,
                       rect.x + 8, rect.y + rect.h / 2);
    SDL_RenderDrawLine(ctx.renderer, rect.x + 4, rect.y + rect.h / 2 + 3,
                       rect.x + 8, rect.y + rect.h / 2);
    SDL_RenderDrawLine(ctx.renderer, rect.x + rect.w - 4, rect.y + rect.h / 2 - 3,
                       rect.x + rect.w - 8, rect.y + rect.h / 2);
    SDL_RenderDrawLine(ctx.renderer, rect.x + rect.w - 4, rect.y + rect.h / 2 + 3,
                       rect.x + rect.w - 8, rect.y + rect.h / 2);
  }
};

struct UIPanel : UIWidget {
  std::string title;
  SDL_Color bg{20,28,40,235};
  SDL_Color border{80,100,140,255};
  SDL_Color title_color{200,220,240,255};
  std::vector<UIWidget*> children;

  void add(UIWidget *w) { children.push_back(w); }

  void handle(UIContext &ctx) override {
    if (!visible) return;
    for (auto *w : children) w->handle(ctx);
  }

  void draw(UIContext &ctx) override {
    if (!visible) return;
    SDL_SetRenderDrawColor(ctx.renderer, bg.r, bg.g, bg.b, bg.a);
    SDL_RenderFillRect(ctx.renderer, &rect);
    SDL_SetRenderDrawColor(ctx.renderer, border.r, border.g, border.b, border.a);
    SDL_RenderDrawRect(ctx.renderer, &rect);

    if (!title.empty()) {
      ui_render_text(ctx.renderer, ctx.font, title.c_str(), title_color,
                     rect.x + 8, rect.y + 4);
    }
    for (auto *w : children) w->draw(ctx);
  }
};
