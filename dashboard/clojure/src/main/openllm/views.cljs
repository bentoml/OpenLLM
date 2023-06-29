(ns openllm.views
  (:require [re-frame.core :as rf]
            [openllm.db :as db])) ;; TODO: remove this. just for the standard llm-config for now

(def icon-path-4-bars "M4 6h16M4 10h16M4 14h16M4 18h16")
(def icon-path-house "M12 20v-6h4v6h5v-8h3L12 3 1 12h3v8z")

(defn second-page
  []
  [:div {:class "min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8"}
   [:div {:class "mt-8 sm:mx-auto sm:w-full sm:max-w-md"}
    [:div {:class "bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10"}
     [:h2 {:class "mt-6 text-center text-3xl font-extrabold text-gray-900"} "Another page"]]]])

(defn openllm-tag
  "The 'OpenLLM' tag in the very top left corner of the screen."
  []
  [:div {:class "flex items-center flex-shrink-0 px-6"}
   [:div {:class "text-center text-3xl font-bold text-blue-1000"} "OpenLLM"]])

(defn sidebar-group-headline
  "The headlines for the different groups in the sidebar are rendered using this component."
  [headline]
  [:h3 {:class "px-3 text-xs font-semibold text-gray-500 uppercase tracking-wider" :id (str "sidebar-headline-" headline)} headline])

(defn selected-model-data
  "The part of the dropdown menu that is actually being displayed."
  []
  [:span {:class "flex min-w-0 items-center justify-between space-x-3"}
   [:img {:class "w-10 h-10 bg-gray-300 rounded-full flex-shrink-0"
          :src "https://www.promptx.ai/assets/images/promptxai-logo-256.png" :alt ""}]
   [:span {:class "flex-1 min-w-0 text-left"}
    [:span {:class "text-gray-900 text-sm font-medium truncate"} "FLAN-T5"]
    [:br]
    [:span {:class "text-gray-500 text-sm truncate"} "flan-t5-large"]]])

(defn model-dropdown-collapsed
  "The whole drop down menu where models are selected by the user. At the core there is a button
   that is used to toggle the dropdown menu."
  []
  [:div
   [sidebar-group-headline "Select model"]
   [:button {:type "button"
             :on-click #(rf/dispatch [:toggle-model-dropdown])
             :class "group w-full bg-gray-100 rounded-md mt-2 px-3.5 py-2 text-sm font-medium text-gray-700 hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-gray-100 focus:ring-pink-500"
             :id "options-menu"
             :aria-expanded "false"
             :aria-haspopup "true"}
    [:span {:class "flex w-full justify-between items-center"}
     [selected-model-data]
     [:svg {:class "flex-shrink-0 h-5 w-5 text-gray-400 group-hover:text-gray-500" :xmlns "http://www.w3.org/2000/svg" :viewBox "0 0 20 20" :fill "currentColor" :aria-hidden "true"}
      [:path {:fill-rule "evenodd" :d "M10 3a1 1 0 01.707.293l3 3a1 1 0 01-1.414 1.414L10 5.414 7.707 7.707a1 1 0 01-1.414-1.414l3-3A1 1 0 0110 3zm-3.707 9.293a1 1 0 011.414 0L10 14.586l2.293-2.293a1 1 0 011.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" :clip-rule "evenodd"}]]]]])

(defn model-dropdown-item-container
  "Renders if the dropdown is expanded. It contains all the models that the user can select."
  []
  (let [dropdown-active? (rf/subscribe [:model-dropdown-active?])]
    (fn model-dropdown-item-container []
      (when @dropdown-active?
        [:div {:class "z-10 mx-3 origin-top absolute right-0 left-0 mt-1 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 divide-y divide-gray-200 focus:outline-none" :role "menu" :aria-orientation "vertical" :aria-labelledby "options-menu"}
         [:div {:class "py-1" :role "none"}
          [:a {:href "#" :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"} "This should look like above"]
          [:a {:href "#" :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"} "And there should be models"]
          [:a {:href "#" :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"} "bla bla"]]
         [:div {:class "py-1" :role "none"}
          [:a {:href "#"
               :on-click #(rf/dispatch [:set-screen-id :second-page])
               :class "block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 hover:text-gray-900" :role "menuitem"}
           "Change View"]]]))))


(defn model-dropdown
  "Aggregates the two components making up the dropdown menu."
  []
  [:div {:class "px-3 mt-8 relative inline-block text-left"}
   [model-dropdown-collapsed]
   [model-dropdown-item-container]])

(defn option-button
  "A button from the status bar."
  [text icon-path]
  [:a {:href "#" :class "text-gray-700 hover:text-gray-900 hover:bg-gray-50 group flex items-center px-2 py-2 text-sm font-medium rounded-md"}
   [:svg {:class "text-gray-400 group-hover:text-gray-500 mr-3 h-6 w-6" :xmlns "http://www.w3.org/2000/svg" :fill "none" :viewBox "0 0 24 24" :stroke "currentColor" :aria-hidden "true"}
    [:path {:stroke-linecap "round" :stroke-linejoin "round" :stroke-width "2" :d icon-path}]] text])

(defn navigation-elements
  "The navigation elements in the sidebar."
  []
  [:div {:class "mt-8"}
   [:div {:class "px-3 mt-8"}
    [sidebar-group-headline "Model options (?)"]]
   [:nav {:class "end-96 px-2 mt-1"}
    [:div {:class "space-y-1"}
     [option-button "Another view" icon-path-house]
     [option-button "I do not know" icon-path-4-bars]]]])

(defn status-display
  "Displays the current service status at the bottom of the sidebar."
  [status-good?]
  (let [status-color (if status-good? "bg-green-500" "bg-red-500")
        status-text (if status-good? "Operational" "Degraded Performance")]
    [:div {:class "px-3 mt-6"}
     [:div {:class "mt-8"}
      [sidebar-group-headline "Service Status"]
      [:div {:class "space-y-1" :role "group" :aria-labelledby "service-status-headline"}
       [:a {:href "#" :class "group flex items-center px-3 py-1 text-sm font-medium text-gray-700 rounded-md"}
        [:span {:class (str "w-2.5 h-2.5 mr-4 rounded-full " status-color) :aria-hidden "true"}]
        [:span {:class "truncate"} status-text]]]]]))

(defn side-bar
  "The render function of the toolbar on the very left of the screen"
  []
  [:div {:class "flex flex-col w-80 border-r border-gray-200 pt-5 pb-4 bg-gray-200"} ;; sidebar div + background
   [openllm-tag]
   [:div {:class "h-0 flex-1 flex flex-col overflow-y-auto"}
    [model-dropdown]
    [navigation-elements]]
   [status-display true]])

(defn chat-controls
  "The chat input field and the send button."
  []
  (let [chat-input-sub (rf/subscribe [:chat-input-value])
        on-change #(rf/dispatch [:set-chat-input-value (.. % -target -value)])
        on-send-click #(rf/dispatch [:on-send-button-click @chat-input-sub db/standard-llm-config])]
    (fn chat-controls
      []
      [:div {:class "fixed bottom-0 px-4 py-2 mt-6 w-full"}
       [:form {:class "flex items-center justify-between"
               :on-submit #(do % (on-send-click)
                               (.preventDefault %))}
        [:input {:class "py-1 w-[calc(100%_-_80px)] appearance-none block border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm"
                 :type "text" :placeholder "Type your message..."
                 :value @chat-input-sub
                 :on-change on-change
                 :id "chat-input"
                 :autocomplete "off"
                 :autocorrect "off"}]
        [:button {:class "ml-2 px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none"
                  :on-click on-send-click
                  :type "button"} "Send"]]])))

(defn chat-history
  "The chat history."
  []
  (let [history (rf/subscribe [:chat-history])]
    (fn chat-history []
      (into [:div {:class "flex-1 overflow-auto mt-6"}]
            (map (fn [{:keys [user text]}]
                   (let [diplay-user (if (= user :model) "Model" "You")
                         color (if (= user :model) "bg-gray-200" "bg-blue-200")]
                     [:div {:class (str "p-2 rounded-lg mb-2 " color)}
                      [:h3 {:class "font-bold text-lg"} diplay-user]
                      [:p {:class "text-gray-700"} text]]))
                 @history)))))

(defn dashboard
  []
  [:div {:class "h-screen flex overflow-hidden bg-white"}
   [:div {:class "hidden lg:flex lg:flex-shrink-0"}
    [side-bar]]
   [:div {:class "flex flex-col w-0 flex-1 overflow-hidden"}
    [:main {:class "flex-1 relative z-0 overflow-y-auto focus:outline-none" :tabIndex "0"}
     [:div {:class "px-4 mt-6 sm:px-6 lg:px-8"}
      [:h2 {:class "text-gray-500 text-xs font-medium uppercase tracking-wide"} "Chat"]
      [chat-history]]
     [chat-controls]]]])
