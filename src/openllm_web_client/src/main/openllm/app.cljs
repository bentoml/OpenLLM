(ns openllm.app
    (:require [reagent.dom :as dom]
              [openllm.views :as views]
              [re-frame.core :as rf]

              ;; the following are only required to make the compiler load the namespaces
              [openllm.events]
              [openllm.subs]
              ))

(def screen->view
  "This map acts the central router for now. This the values of this map are the
   render functions and the keys are clojure keywords that are used to identify
   the screen that should be rendered."
  {:main [views/dashboard]
   :second-page [views/second-page]})

(defn app
  []
  (let [screen-id @(rf/subscribe [:screen-id])]
    (screen->view screen-id)))

(defn ^:dev/after-load start []
    (dom/render [app]
        (.getElementById js/document "app")))

(defn init
  "This init function is called exactly once when the page loads."
  []
  (enable-console-print!) ;; so that println writes to `console.log`
  (rf/dispatch-sync [:initialise-db])
  (start))
