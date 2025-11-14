import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Trash2, Edit } from "lucide-react";
import { format } from "date-fns";
import { es } from "date-fns/locale";

export default function SwimTimesTable({ times, swimmers = [], onDelete, onEdit, showActions = false }) {
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = (seconds % 60).toFixed(2);
    return mins > 0 ? `${mins}:${secs.padStart(5, '0')}` : `${secs}s`;
  };

  const getSwimmerName = (swimmerId) => {
    const swimmer = swimmers.find(s => s.id === swimmerId);
    return swimmer ? swimmer.name : 'Desconocido';
  };

  if (times.length === 0) {
    return (
      <div className="text-center py-12 text-gray-500" data-testid="no-times-message">
        <p className="text-lg">No hay tiempos registrados aún</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow className="bg-gray-50">
            {swimmers.length > 0 && <TableHead className="font-semibold">Nadador</TableHead>}
            <TableHead className="font-semibold">Distancia</TableHead>
            <TableHead className="font-semibold">Estilo</TableHead>
            <TableHead className="font-semibold">Tiempo</TableHead>
            <TableHead className="font-semibold">Fecha</TableHead>
            <TableHead className="font-semibold">Competición</TableHead>
            {showActions && <TableHead className="font-semibold text-right">Acciones</TableHead>}
          </TableRow>
        </TableHeader>
        <TableBody>
          {times.map((time) => (
            <TableRow key={time.id} className="hover:bg-gray-50" data-testid={`time-row-${time.id}`}>
              {swimmers.length > 0 && (
                <TableCell className="font-medium">{getSwimmerName(time.swimmer_id)}</TableCell>
              )}
              <TableCell>{time.distance}m</TableCell>
              <TableCell>
                <span className="px-3 py-1 rounded-full text-sm font-medium bg-[#278D33]/10 text-[#278D33]">
                  {time.style}
                </span>
              </TableCell>
              <TableCell className="font-mono font-bold text-[#278D33]">{formatTime(time.time_seconds)}</TableCell>
              <TableCell>{format(new Date(time.date), 'dd MMM yyyy', { locale: es })}</TableCell>
              <TableCell className="text-gray-600">{time.competition || '-'}</TableCell>
              {showActions && (
                <TableCell className="text-right">
                  <div className="flex justify-end gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onEdit(time)}
                      className="text-[#278D33] hover:text-[#1f6b28] hover:bg-[#278D33]/10"
                      data-testid={`edit-time-${time.id}`}
                    >
                      <Edit className="w-4 h-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => onDelete(time.id)}
                      className="text-red-600 hover:text-red-700 hover:bg-red-50"
                      data-testid={`delete-time-${time.id}`}
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                </TableCell>
              )}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}
