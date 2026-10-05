# Лабораторная работа № 1 — вариант 3: параллелепипед

Исходный проект преподавателя дополнен сценой Vulkan. Есть перспектива и ортографическая проекция, управление переносом, вращением и масштабом, анимация с паузой и настройками, цвет в ImGui, процедурные цвета вершин и второй экземпляр фигуры с отдельным `VkDescriptorSet`. Используются Vulkan, GLFW, ImGui и C++20.

## Как собрать на Windows

1. Установите Visual Studio 2022 с компонентом «Разработка классических приложений на C++», CMake и [Vulkan SDK](https://vulkan.lunarg.com/sdk/home). В составе SDK нужен `glslc`; после установки откройте новый терминал и проверьте `glslc --version`.
2. Откройте терминал в каталоге, где лежит этот файл. Выполните `cmake --preset msvc-debug`, затем `cmake --build build-debug --config Debug`.
3. Из этого же каталога запустите `build-debug/Debug/vulkan-starter-app.exe`. Для первого запуска нужен доступ CMake к GitHub: `FetchContent` скачивает зависимости преподавателя. Путь к шейдерам задаёт CMake, поэтому запуск из Visual Studio и из терминала работает одинаково.
4. Если в Visual Studio проект уже открыт, скопируйте `source/` и `shaders/` в ваш клон, замените `CMakeLists.txt`, обновите конфигурацию и соберите. Оставшиеся исходники преподавателя уже включены здесь.

Без Vulkan SDK, `glslc` и загруженных зависимостей сборка невозможна. Если окно открылось, проверьте переключатель Perspective projection, движение ползунков Position/Rotation/Scale, Pause animation, Orbit speed/Orbit radius, Color multiplier и Second object. В консоли не должно быть сообщений validation layers.

