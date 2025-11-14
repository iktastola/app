import { Shirt, ShoppingBag } from "lucide-react";

export default function LockerView({ locker }) {
  if (!locker) {
    return (
      <div className="text-center py-12 text-gray-500" data-testid="no-locker-message">
        <ShoppingBag className="w-16 h-16 mx-auto mb-4 opacity-30" />
        <p className="text-lg">Aún no se ha asignado tu taquilla</p>
        <p className="text-sm mt-2">Contacta con el administrador del club</p>
      </div>
    );
  }

  const items = [
    { name: 'Pantalón', size: locker.pants_size, icon: '👖' },
    { name: 'Camiseta', size: locker.shirt_size, icon: '👕' },
    { name: 'Sudadera', size: locker.hoodie_size, icon: '🧥' },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6" data-testid="locker-items">
      {items.map((item, index) => (
        <div
          key={index}
          className="bg-gradient-to-br from-emerald-50 to-teal-50 p-6 rounded-xl border-2 border-[#278D33]/20 hover:border-[#278D33]/40 transition-all card-hover"
          data-testid={`locker-item-${index}`}
        >
          <div className="text-center">
            <div className="text-5xl mb-4">{item.icon}</div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">{item.name}</h3>
            <div className="inline-flex items-center gap-2 bg-white px-4 py-2 rounded-full shadow-sm">
              <Shirt className="w-4 h-4 text-[#278D33]" />
              <span className="font-semibold text-[#278D33] text-lg">Talla {item.size}</span>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
