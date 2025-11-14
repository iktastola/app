# 🏊‍♂️ Club de Natación Astolai - Credenciales de Acceso

## 📋 Usuarios de Prueba

### 👨‍💼 Administrador
- **Email:** admin@astolai.com
- **Contraseña:** admin123
- **Permisos:** Gestión completa de usuarios y taquillas

### 👨‍🏫 Entrenador
- **Email:** entrenador@astolai.com
- **Contraseña:** coach123
- **Permisos:** Ver y editar tiempos de todos los nadadores

### 🏊‍♀️ Nadadores

#### Ana García
- **Email:** ana@astolai.com
- **Contraseña:** nadador123
- **Nota:** Tiene tiempos y taquilla asignada

#### Pablo López
- **Email:** pablo@astolai.com
- **Contraseña:** nadador123
- **Nota:** Usuario sin tiempos registrados

#### María Fernández
- **Email:** maria@astolai.com
- **Contraseña:** nadador123
- **Nota:** Usuario sin tiempos registrados

---

## 🎯 Funcionalidades por Rol

### Nadadores
- ✅ Ver sus propios tiempos de natación
- ✅ Ver sus mejores marcas personales (agrupadas por estilo)
- ✅ Ver su taquilla virtual (pantalón, camiseta, sudadera con tallas)

### Entrenadores
- ✅ Ver todos los tiempos de todos los nadadores
- ✅ Registrar nuevos tiempos (con distancia, estilo, tiempo, fecha, competición)
- ✅ Editar tiempos existentes
- ✅ Eliminar tiempos
- ✅ Filtrar tiempos por nadador

### Administradores
- ✅ Crear nuevos usuarios (nadadores, entrenadores, administradores)
- ✅ Ver lista completa de usuarios
- ✅ Eliminar usuarios
- ✅ Gestionar taquillas (asignar tallas de ropa a nadadores)

---

## 🏊 Datos del Club

### Estilos de Natación
- Libre
- Espalda
- Braza
- Mariposa

### Distancias
- 50m
- 100m
- 200m
- 400m
- 800m
- 1500m

### Tallas Disponibles
- XS, S, M, L, XL, XXL

---

## 🎨 Diseño

La aplicación utiliza los colores corporativos del Club de Natación Astolai:
- **Color principal:** #278D33 (verde corporativo)
- **Tipografía:** Space Grotesk (títulos) e Inter (texto)
- **Estilo:** Moderno, limpio y deportivo con animaciones suaves

---

## 📱 URL de Acceso

**Frontend:** https://swimteam-hub.preview.emergentagent.com

---

## 🔧 Base de Datos

- **Base de datos:** MongoDB
- **Nombre de BD:** swim_club_db

### Colecciones:
1. **users** - Información de usuarios (nadadores, entrenadores, administradores)
2. **swim_times** - Tiempos de natación registrados
3. **personal_bests** - Mejores marcas personales (calculadas automáticamente)
4. **lockers** - Taquillas virtuales con tallas de ropa
