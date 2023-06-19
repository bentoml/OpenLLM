(ns tailwind.app
    (:require [reagent.dom :as dom]
              [tailwind.views :as views]
              [tailwind.db :as db]))

(def screen->view
  "This map acts the central router for now. This the values of this map are the
   render functions and the keys are clojure keywords that are used to identify
   the screen that should be rendered."
  {:main [views/dashboard]
   :second-page [views/second-page]})

(defn app
  []
  (screen->view :main)
  #_(:screen-id @db/state))

(defn ^:dev/after-load start []
    (dom/render [app]
        (.getElementById js/document "app")))

(defn init
  "This init function is called exactly once when the page loads."
  []
  (start))
