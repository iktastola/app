import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Users } from "lucide-react";

export default function SwimmerSelector({ swimmers, selectedSwimmer, onSelectSwimmer }) {
  return (
    <div className="w-full sm:w-64">
      <Select value={selectedSwimmer || "all"} onValueChange={(value) => onSelectSwimmer(value === "all" ? null : value)}>
        <SelectTrigger className="border-[#278D33]/30 focus:ring-[#278D33]" data-testid="swimmer-selector">
          <Users className="w-4 h-4 mr-2 text-[#278D33]" />
          <SelectValue placeholder="Todos los nadadores" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="all">Todos los nadadores</SelectItem>
          {swimmers.map((swimmer) => (
            <SelectItem key={swimmer.id} value={swimmer.id} data-testid={`swimmer-option-${swimmer.id}`}>
              {swimmer.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
