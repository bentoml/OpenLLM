(ns openllm.app
    (:require [openllm.api.core :as api]
              [openllm.api.indexed-db.core :as idb]
              [reagent.dom :as dom]
              [openllm.views :as views]
              [re-frame.core :as rf]

              ;; the following are only required to make the compiler load the namespaces
              [day8.re-frame.http-fx]
              [openllm.events]
              [openllm.subs]
              [openllm.api.events]))

(defn app
  []
  [views/dashboard])

(defn ^:dev/after-load start []
  (dom/render [app]
              (.getElementById js/document "app")))

(defn init
  "This init function is called exactly once when the page loads."
  []
  (enable-console-print!) ;; so that print writes to `console.log`
  (rf/dispatch-sync [:initialise-db])
  (api/init-idb)
  (rf/dispatch-sync [::set-idb idb/idb])
  (start))
